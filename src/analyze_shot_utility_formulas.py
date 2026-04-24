#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ablate shot-level pseudo-utility formulas before training.

This is Phase 1b. It is deliberately read-only with respect to the training
pipeline:

- It does not train a model.
- It does not modify current sparse-pair training.
- It does not write checkpoints.
- It does not use gtscore/user_summary to construct any formula.
- It uses gtscore only for offline diagnostics.

Input:
    pseudo_labels/{dataset}/shot_utility.npy

This file is produced by:
    src/make_shot_pseudo_utility.py

The script evaluates multiple candidate formulas built from stored components:
    semantic_coverage
    visual_representativeness
    distinctiveness = 1 - visual_representativeness
    redundancy_penalty
    anti_redundancy = 1 - redundancy_penalty
    eventiveness

It reports correlation with shot-level HDF5 gtscore for diagnosis only.

Do not select a final formula using test results. Use this script to inspect
whether there exists a cross-dataset non-reversed utility candidate before
connecting any listwise loss to the training loop.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch

from anchor_free.train_mil_cond import build_sampled_to_shot_overlap
from helpers.eval_protocol_helper import (
    compute_kendall_tau_b,
    compute_spearman_rho,
    safe_nanmean,
)
from helpers.mil_data_helper_cond import VideoDatasetMILCond
from helpers.mil_path_helper import get_dataset_pseudo_dir
from run_train_mil_cond import load_all_splits, validate_splits


logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Diagnose candidate shot utility formulas.'
    )
    parser.add_argument('--dataset', type=str, required=True, choices=('summe', 'tvsum'))
    parser.add_argument('--splits', type=str, nargs='+', required=True)

    parser.add_argument('--text-cond-num', type=int, default=10)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=12345)

    parser.add_argument(
        '--split-parts',
        type=str,
        nargs='+',
        default=('train', 'val', 'test'),
        choices=('train', 'val', 'test'),
        help='Which split partitions to include in the overall analysis.',
    )
    parser.add_argument(
        '--selection-part',
        type=str,
        default='val',
        choices=('train', 'val', 'test', 'all'),
        help='Partition used to rank formulas in the printed recommendation. '
             'This is diagnostic only; do not use test to tune a paper result.',
    )

    parser.add_argument(
        '--shot-utility-path',
        type=str,
        default=None,
        help='Default: pseudo_labels/{dataset}/shot_utility.npy',
    )
    parser.add_argument('--max-videos', type=int, default=None)

    parser.add_argument(
        '--out-json',
        type=str,
        default=None,
        help='Default: diagnostics/{dataset}_shot_utility_formula_ablation.json',
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        default=None,
        help='Default: diagnostics/{dataset}_shot_utility_formula_ablation.csv',
    )
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='[%(levelname)s] %(message)s',
    )


def normalize_01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    if not np.isfinite(values).all():
        raise ValueError('normalize_01 received non-finite values.')
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo + eps)).astype(np.float32)


def collect_keys_by_part(splits: List[Dict]) -> Dict[str, List[str]]:
    out = {'train': [], 'val': [], 'test': []}
    for split in splits:
        out['train'].extend(split['train_keys'])
        out['val'].extend(split['val_keys'])
        out['test'].extend(split['test_keys'])

    # Deterministic dedupe.
    return {part: sorted(set(keys)) for part, keys in out.items()}


def collect_union(parts: Iterable[str], keys_by_part: Dict[str, List[str]]) -> List[str]:
    keys: List[str] = []
    for part in parts:
        keys.extend(keys_by_part[part])
    return sorted(set(keys))


def load_shot_utility(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f'Shot utility file not found: {path}')

    obj = np.load(path, allow_pickle=True)
    try:
        obj = obj.item()
    except Exception as exc:
        raise ValueError(f'Invalid shot utility npy format: {path}') from exc

    if not isinstance(obj, dict):
        raise ValueError(f'Shot utility file must contain dict[h5_key -> dict], got {type(obj)}')

    return obj


def aggregate_sample_scores_to_shots(
    sample_scores: np.ndarray,
    overlaps: torch.Tensor,
    shot_lengths: torch.Tensor,
) -> np.ndarray:
    sample_scores = np.asarray(sample_scores, dtype=np.float32).reshape(-1)
    scores_t = torch.tensor(sample_scores, dtype=torch.float32, device=overlaps.device)

    if overlaps.shape[1] != scores_t.shape[0]:
        raise ValueError(
            f'Score length mismatch: overlaps={tuple(overlaps.shape)} vs scores={tuple(scores_t.shape)}'
        )

    shot_scores = torch.matmul(overlaps, scores_t) / shot_lengths.clamp_min(1.0)
    return shot_scores.detach().cpu().numpy().astype(np.float32)


def safe_corr(pred: np.ndarray, target: np.ndarray, key: str) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if pred.shape != target.shape or pred.size < 2:
        return float('nan'), float('nan')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        return float('nan'), float('nan')
    if float(pred.max() - pred.min()) == 0.0 or float(target.max() - target.min()) == 0.0:
        return float('nan'), float('nan')

    try:
        tau = compute_kendall_tau_b(pred, target, key=key)
    except Exception:
        tau = float('nan')

    try:
        rho = compute_spearman_rho(pred, target, key=key)
    except Exception:
        rho = float('nan')

    return float(tau), float(rho)


def get_component(record: Dict, name: str) -> np.ndarray:
    if name not in record:
        raise KeyError(f'Missing component "{name}" in shot utility record.')
    arr = np.asarray(record[name], dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f'Non-finite component "{name}".')
    return arr


def build_components(record: Dict) -> Dict[str, np.ndarray]:
    semantic = get_component(record, 'semantic_coverage')
    representativeness = get_component(record, 'visual_representativeness')
    redundancy = get_component(record, 'redundancy_penalty')
    eventiveness = get_component(record, 'eventiveness')
    phase1_default = get_component(record, 'shot_utility')

    lengths = {
        semantic.shape[0],
        representativeness.shape[0],
        redundancy.shape[0],
        eventiveness.shape[0],
        phase1_default.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError(
            f'Component length mismatch: '
            f'sem={semantic.shape}, rep={representativeness.shape}, '
            f'red={redundancy.shape}, event={eventiveness.shape}, default={phase1_default.shape}'
        )

    return {
        'phase1_default': normalize_01(phase1_default),
        'semantic': normalize_01(semantic),
        'representativeness': normalize_01(representativeness),
        'distinctiveness': normalize_01(1.0 - normalize_01(representativeness)),
        'redundancy': normalize_01(redundancy),
        'anti_redundancy': normalize_01(1.0 - normalize_01(redundancy)),
        'eventiveness': normalize_01(eventiveness),
    }


FormulaFn = Callable[[Dict[str, np.ndarray]], np.ndarray]


def formula_definitions() -> Dict[str, FormulaFn]:
    def n(x):
        return normalize_01(x)

    return {
        # Single components.
        'phase1_default': lambda c: c['phase1_default'],
        'semantic': lambda c: c['semantic'],
        'representativeness': lambda c: c['representativeness'],
        'distinctiveness': lambda c: c['distinctiveness'],
        'anti_redundancy': lambda c: c['anti_redundancy'],
        'eventiveness': lambda c: c['eventiveness'],

        # Conservative two-term candidates.
        'semantic_plus_rep': lambda c: n(c['semantic'] + c['representativeness']),
        'semantic_plus_distinct': lambda c: n(c['semantic'] + c['distinctiveness']),
        'semantic_plus_anti_redundancy': lambda c: n(c['semantic'] + c['anti_redundancy']),
        'rep_plus_anti_redundancy': lambda c: n(c['representativeness'] + c['anti_redundancy']),
        'distinct_plus_anti_redundancy': lambda c: n(c['distinctiveness'] + c['anti_redundancy']),

        # Redundancy-subtracted variants.
        'semantic_minus_red': lambda c: n(c['semantic'] - 0.2 * c['redundancy']),
        'rep_minus_red': lambda c: n(c['representativeness'] - 0.2 * c['redundancy']),
        'distinct_minus_red': lambda c: n(c['distinctiveness'] - 0.2 * c['redundancy']),
        'semantic_plus_rep_minus_red': lambda c: n(c['semantic'] + 0.5 * c['representativeness'] - 0.2 * c['redundancy']),
        'semantic_plus_distinct_minus_red': lambda c: n(c['semantic'] + 0.5 * c['distinctiveness'] - 0.2 * c['redundancy']),

        # Product forms, useful when coverage and non-redundancy must co-occur.
        'semantic_x_anti_redundancy': lambda c: n(c['semantic'] * c['anti_redundancy']),
        'rep_x_anti_redundancy': lambda c: n(c['representativeness'] * c['anti_redundancy']),
        'semantic_x_rep': lambda c: n(c['semantic'] * c['representativeness']),
        'semantic_x_distinct': lambda c: n(c['semantic'] * c['distinctiveness']),

        # Eventiveness as weak auxiliary, not default.
        'semantic_plus_event': lambda c: n(c['semantic'] + 0.25 * c['eventiveness']),
        'semantic_plus_rep_plus_event': lambda c: n(c['semantic'] + 0.5 * c['representativeness'] + 0.25 * c['eventiveness']),
        'semantic_plus_distinct_plus_event': lambda c: n(c['semantic'] + 0.5 * c['distinctiveness'] + 0.25 * c['eventiveness']),
        'semantic_plus_event_minus_red': lambda c: n(c['semantic'] + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']),
        'semantic_plus_rep_plus_event_minus_red': lambda c: n(c['semantic'] + 0.5 * c['representativeness'] + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']),
        'semantic_plus_distinct_plus_event_minus_red': lambda c: n(c['semantic'] + 0.5 * c['distinctiveness'] + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']),
    }


def compute_shot_gtscore(dataset: VideoDatasetMILCond, index: int) -> Tuple[str, np.ndarray]:
    (
        key,
        seq,
        soft_label,
        text_cond,
        text_target,
        all_text_features,
        caption_spans_idx,
        caption_valid_mask,
        gtscore,
        user_summary,
        cps,
        n_frames,
        nfps,
        picks,
    ) = dataset[index]

    if gtscore is None:
        raise ValueError(f'Missing gtscore for key: {key}')

    picks_np = np.asarray(picks, dtype=np.int32)
    gtscore_np = np.asarray(gtscore, dtype=np.float32).reshape(-1)

    if gtscore_np.shape[0] != picks_np.shape[0]:
        raise ValueError(
            f'gtscore/picks length mismatch for {key}: {gtscore_np.shape[0]} vs {picks_np.shape[0]}'
        )

    overlaps, shot_lengths = build_sampled_to_shot_overlap(
        picks=torch.tensor(picks_np, dtype=torch.long),
        cps=torch.tensor(np.asarray(cps, dtype=np.int32), dtype=torch.long),
        n_frames=int(np.asarray(n_frames).item()),
    )

    shot_gtscore = aggregate_sample_scores_to_shots(
        sample_scores=gtscore_np,
        overlaps=overlaps,
        shot_lengths=shot_lengths,
    )

    return str(key), shot_gtscore


def evaluate_formula_records(
    dataset: VideoDatasetMILCond,
    shot_utility_by_key: Dict[str, Dict],
    formula_fns: Dict[str, FormulaFn],
) -> List[Dict]:
    rows: List[Dict] = []

    for idx in range(len(dataset)):
        key, shot_gtscore = compute_shot_gtscore(dataset, idx)
        h5_key = Path(key).name

        if h5_key not in shot_utility_by_key:
            raise KeyError(f'Missing h5 key in shot utility file: {h5_key}')

        record = shot_utility_by_key[h5_key]
        components = build_components(record)

        for formula_name, formula_fn in formula_fns.items():
            try:
                scores = normalize_01(formula_fn(components))
            except Exception as exc:
                raise RuntimeError(f'Failed formula "{formula_name}" for {h5_key}') from exc

            if scores.shape[0] != shot_gtscore.shape[0]:
                raise ValueError(
                    f'Formula/gtscore length mismatch for {h5_key}, formula={formula_name}: '
                    f'{scores.shape[0]} vs {shot_gtscore.shape[0]}'
                )

            tau, rho = safe_corr(scores, shot_gtscore, key=f'{key}:{formula_name}')

            rows.append({
                'key': key,
                'h5_key': h5_key,
                'formula': formula_name,
                'num_shots': int(scores.shape[0]),
                'score_std': float(np.std(scores)) if scores.size else 0.0,
                'kendall': tau,
                'spearman': rho,
            })

    return rows


def summarize_formula_rows(rows: List[Dict], keys: List[str]) -> Dict[str, Dict]:
    key_set = {str(k) for k in keys}
    filtered = [r for r in rows if r['key'] in key_set]

    formulas = sorted(set(r['formula'] for r in filtered))
    out: Dict[str, Dict] = {}

    for formula in formulas:
        sub = [r for r in filtered if r['formula'] == formula]
        kendalls = [float(r['kendall']) for r in sub if np.isfinite(float(r['kendall']))]
        spearmans = [float(r['spearman']) for r in sub if np.isfinite(float(r['spearman']))]
        score_stds = [float(r['score_std']) for r in sub if np.isfinite(float(r['score_std']))]

        out[formula] = {
            'num_videos': int(len(sub)),
            'valid_kendall_count': int(len(kendalls)),
            'valid_spearman_count': int(len(spearmans)),
            'mean_kendall': safe_nanmean(kendalls, default=float('nan')),
            'median_kendall': float(np.median(kendalls)) if kendalls else float('nan'),
            'positive_kendall_rate': float(np.mean([v > 0 for v in kendalls])) if kendalls else float('nan'),
            'mean_spearman': safe_nanmean(spearmans, default=float('nan')),
            'median_spearman': float(np.median(spearmans)) if spearmans else float('nan'),
            'positive_spearman_rate': float(np.mean([v > 0 for v in spearmans])) if spearmans else float('nan'),
            'mean_score_std': safe_nanmean(score_stds, default=float('nan')),
        }

    return out


def rank_formulas(summary: Dict[str, Dict]) -> List[Dict]:
    ranked = []
    for name, stats in summary.items():
        mean_k = float(stats['mean_kendall'])
        mean_s = float(stats['mean_spearman'])
        pos_rate = float(stats['positive_kendall_rate'])
        score = mean_k + 0.25 * mean_s + 0.05 * pos_rate
        ranked.append({
            'formula': name,
            'score': score,
            'mean_kendall': mean_k,
            'mean_spearman': mean_s,
            'positive_kendall_rate': pos_rate,
            'mean_score_std': float(stats['mean_score_std']),
        })

    ranked.sort(
        key=lambda x: (
            np.nan_to_num(x['score'], nan=-999.0),
            np.nan_to_num(x['mean_kendall'], nan=-999.0),
            np.nan_to_num(x['mean_spearman'], nan=-999.0),
        ),
        reverse=True,
    )
    return ranked


def atomic_save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=True)
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    if not rows:
        tmp.write_text('', encoding='utf-8')
        tmp.replace(path)
        return

    fieldnames = list(rows[0].keys())
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def main() -> None:
    args = get_parser().parse_args()
    setup_logging(args.log_level)

    shot_utility_path = (
        Path(args.shot_utility_path)
        if args.shot_utility_path is not None
        else get_dataset_pseudo_dir(args.dataset) / 'shot_utility.npy'
    )
    out_json = (
        Path(args.out_json)
        if args.out_json is not None
        else Path('diagnostics') / f'{args.dataset}_shot_utility_formula_ablation.json'
    )
    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else Path('diagnostics') / f'{args.dataset}_shot_utility_formula_ablation.csv'
    )

    splits = load_all_splits(
        split_paths=args.splits,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    validate_splits(splits, args.dataset)

    keys_by_part = collect_keys_by_part(splits)
    selected_keys = collect_union(args.split_parts, keys_by_part)
    if args.max_videos is not None:
        selected_keys = selected_keys[:args.max_videos]

    if args.selection_part == 'all':
        recommendation_keys = selected_keys
    else:
        recommendation_keys = sorted(set(keys_by_part[args.selection_part]).intersection(selected_keys))

    logger.info(
        'Formula ablation | dataset=%s | videos=%d | selection_part=%s | shot_utility=%s',
        args.dataset,
        len(selected_keys),
        args.selection_part,
        shot_utility_path,
    )

    shot_utility_by_key = load_shot_utility(shot_utility_path)
    dataset = VideoDatasetMILCond(
        keys=selected_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
    )

    formulas = formula_definitions()
    rows = evaluate_formula_records(
        dataset=dataset,
        shot_utility_by_key=shot_utility_by_key,
        formula_fns=formulas,
    )

    summary_all = summarize_formula_rows(rows, selected_keys)
    summary_train = summarize_formula_rows(rows, keys_by_part['train'])
    summary_val = summarize_formula_rows(rows, keys_by_part['val'])
    summary_test = summarize_formula_rows(rows, keys_by_part['test'])
    summary_selection = summarize_formula_rows(rows, recommendation_keys)

    ranked_selection = rank_formulas(summary_selection)

    payload = {
        'config': {
            'dataset': args.dataset,
            'splits': args.splits,
            'split_parts': list(args.split_parts),
            'selection_part': args.selection_part,
            'text_cond_num': args.text_cond_num,
            'val_ratio': args.val_ratio,
            'seed': args.seed,
            'shot_utility_path': str(shot_utility_path),
            'num_selected_keys': len(selected_keys),
            'num_recommendation_keys': len(recommendation_keys),
        },
        'formula_names': sorted(formulas.keys()),
        'summary_all': summary_all,
        'summary_train': summary_train,
        'summary_val': summary_val,
        'summary_test': summary_test,
        'summary_selection': summary_selection,
        'ranked_selection': ranked_selection,
        'records': rows,
    }

    atomic_save_json(out_json, payload)
    write_csv(out_csv, rows)

    logger.info('Top formulas on selection_part=%s:', args.selection_part)
    for item in ranked_selection[:10]:
        logger.info(
            '%s | score=%.4f | tau=%.4f | rho=%.4f | pos_tau=%.3f | std=%.4f',
            item['formula'],
            item['score'],
            item['mean_kendall'],
            item['mean_spearman'],
            item['positive_kendall_rate'],
            item['mean_score_std'],
        )

    logger.info('Wrote JSON: %s', out_json)
    logger.info('Wrote CSV: %s', out_csv)


if __name__ == '__main__':
    main()
