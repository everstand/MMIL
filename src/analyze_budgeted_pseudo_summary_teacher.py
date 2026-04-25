#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose budgeted pseudo-summary teachers before changing the model.

This is Phase 2B-0. It is deliberately OFFLINE and NON-INTRUSIVE:

- It does not train a model.
- It does not modify DSNetAFMILCond.
- It does not modify current sparse-pair or listwise training.
- It does not write checkpoints.
- It does not use gtscore/user_summary to construct pseudo summaries.
- It uses user_summary/gtscore only for diagnostics.

Purpose
-------
The previous listwise utility experiments improved rank metrics in some cases
but hurt keyshot F1. This script checks whether a shot-level utility can form a
reasonable 15%-budget pseudo-summary before we use it as a selection teacher.

For each candidate formula:
    shot utility [S]
      -> 0/1 knapsack under 15% budget using HDF5 n_frame_per_seg
      -> pseudo binary summary [n_frames]
      -> F1 against user_summary
      -> optional binary-selection rank diagnostics against shot-level gtscore

It also reports confidence-gated supervision counts:
    positive = budget-selected shots
    negative = non-selected shots with utility <= quantile(q_neg)
    ignore   = all remaining shots

Do not use test results to select a formula for a paper result. Use validation
diagnostics for method selection and test diagnostics only as a sanity check.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from anchor_free.train_mil_cond import build_sampled_to_shot_overlap
from helpers import vsumm_helper
from helpers.eval_protocol_helper import (
    compute_kendall_tau_b,
    compute_spearman_rho,
    infer_f1_metric_from_key,
    safe_nanmean,
)
from helpers.mil_data_helper_cond import VideoDatasetMILCond
from helpers.mil_path_helper import get_dataset_pseudo_dir
from helpers.shot_utility_helper import ShotUtilityStore, formula_definitions
from run_train_mil_cond import load_all_splits, validate_splits


logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Analyze budgeted pseudo-summary teachers from shot utility formulas.'
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
    )
    parser.add_argument(
        '--selection-part',
        type=str,
        default='val',
        choices=('train', 'val', 'test', 'all'),
        help='Partition used to rank candidate formulas in the recommendation field.',
    )

    parser.add_argument(
        '--shot-utility-path',
        type=str,
        default=None,
        help='Default: pseudo_labels/{dataset}/shot_utility.npy',
    )
    parser.add_argument(
        '--formulas',
        type=str,
        nargs='+',
        default=None,
        help='Formula names to evaluate. Default: all formulas defined by shot_utility_helper.',
    )
    parser.add_argument(
        '--summary-budget',
        type=float,
        default=0.15,
        help='Summary length ratio. Default matches canonical SumMe/TVSum keyshot budget.',
    )
    parser.add_argument(
        '--negative-quantile',
        type=float,
        default=0.25,
        help='Utility quantile for confidence-gated negative shots.',
    )
    parser.add_argument('--max-videos', type=int, default=None)

    parser.add_argument(
        '--out-json',
        type=str,
        default=None,
        help='Default: diagnostics/{dataset}_budgeted_pseudo_summary_teacher.json',
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        default=None,
        help='Default: diagnostics/{dataset}_budgeted_pseudo_summary_teacher.csv',
    )
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='[%(levelname)s] %(message)s',
    )


def collect_keys_by_part(splits: List[Dict]) -> Dict[str, List[str]]:
    keys_by_part = {'train': [], 'val': [], 'test': []}
    for split in splits:
        keys_by_part['train'].extend(split['train_keys'])
        keys_by_part['val'].extend(split['val_keys'])
        keys_by_part['test'].extend(split['test_keys'])
    return {k: sorted(set(v)) for k, v in keys_by_part.items()}


def collect_union(parts: Iterable[str], keys_by_part: Dict[str, List[str]]) -> List[str]:
    keys: List[str] = []
    for part in parts:
        keys.extend(keys_by_part[part])
    return sorted(set(keys))


def normalize_01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('normalize_01 received non-finite values.')
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo + eps)).astype(np.float32)


def shot_utility_to_budgeted_summary(
    shot_utility: np.ndarray,
    cps: np.ndarray,
    nfps: np.ndarray,
    n_frames: int,
    budget_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    shot_utility = normalize_01(shot_utility)
    cps = np.asarray(cps, dtype=np.int32)
    nfps = np.asarray(nfps, dtype=np.int32).reshape(-1)

    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {cps.shape}')
    if nfps.shape[0] != cps.shape[0]:
        raise ValueError(f'nfps/cps length mismatch: {nfps.shape[0]} vs {cps.shape[0]}')
    if shot_utility.shape[0] != cps.shape[0]:
        raise ValueError(
            f'shot_utility/cps length mismatch: {shot_utility.shape[0]} vs {cps.shape[0]}'
        )
    if n_frames <= 0:
        raise ValueError(f'Invalid n_frames: {n_frames}')
    if not (0.0 < budget_ratio < 1.0):
        raise ValueError(f'Invalid budget_ratio={budget_ratio}; expected 0 < ratio < 1.')

    # Integer values are required by OR-Tools knapsack. Scale after normalization.
    values = np.round(shot_utility * 1000.0).astype(np.int32)
    capacity = int(n_frames * budget_ratio)

    # If all values are zero, no shot has positive teacher utility.
    if values.size == 0 or int(values.max()) <= 0:
        selected_idx = []
    else:
        selected_idx = vsumm_helper.knapsack(values.tolist(), nfps.tolist(), capacity)

    selected_mask = np.zeros(cps.shape[0], dtype=bool)
    selected_mask[selected_idx] = True

    summary = np.zeros(n_frames, dtype=bool)
    for shot_idx in np.where(selected_mask)[0]:
        first, last = cps[shot_idx]
        first = int(max(0, min(first, n_frames - 1)))
        last = int(max(first, min(last, n_frames - 1)))
        summary[first:last + 1] = True

    return selected_mask, summary


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


def compute_confidence_gate_counts(
    shot_utility: np.ndarray,
    selected_mask: np.ndarray,
    negative_quantile: float,
) -> Dict[str, float]:
    if not (0.0 < negative_quantile < 1.0):
        raise ValueError(f'Invalid negative_quantile={negative_quantile}; expected 0 < q < 1.')

    shot_utility = normalize_01(shot_utility)
    selected_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
    if shot_utility.shape[0] != selected_mask.shape[0]:
        raise ValueError('utility/selected_mask length mismatch.')

    thr_neg = float(np.quantile(shot_utility, negative_quantile))
    positive = selected_mask
    negative = (~selected_mask) & (shot_utility <= thr_neg)
    ignore = ~(positive | negative)

    num_shots = int(shot_utility.shape[0])
    return {
        'num_positive_shots': int(positive.sum()),
        'num_negative_shots': int(negative.sum()),
        'num_ignore_shots': int(ignore.sum()),
        'positive_shot_rate': float(positive.sum() / max(num_shots, 1)),
        'negative_shot_rate': float(negative.sum() / max(num_shots, 1)),
        'ignore_shot_rate': float(ignore.sum() / max(num_shots, 1)),
        'negative_threshold': float(thr_neg),
    }


def evaluate_one_video_formulas(
    dataset: VideoDatasetMILCond,
    index: int,
    utility_store: ShotUtilityStore,
    formulas: List[str],
    budget_ratio: float,
    negative_quantile: float,
) -> List[Dict]:
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

    h5_key = Path(key).name
    n_frames_int = int(np.asarray(n_frames).item())
    cps_np = np.asarray(cps, dtype=np.int32)
    nfps_np = np.asarray(nfps, dtype=np.int32)
    picks_np = np.asarray(picks, dtype=np.int32)

    if user_summary is None:
        raise ValueError(f'Missing user_summary for key: {key}')
    if gtscore is None:
        raise ValueError(f'Missing gtscore for key: {key}')

    eval_metric = infer_f1_metric_from_key(str(key))

    overlaps, shot_lengths = build_sampled_to_shot_overlap(
        picks=torch.tensor(picks_np, dtype=torch.long),
        cps=torch.tensor(cps_np, dtype=torch.long),
        n_frames=n_frames_int,
    )
    shot_gtscore = aggregate_sample_scores_to_shots(
        sample_scores=np.asarray(gtscore, dtype=np.float32),
        overlaps=overlaps,
        shot_lengths=shot_lengths,
    )

    rows: List[Dict] = []

    for formula in formulas:
        utility = utility_store.get(h5_key=h5_key, formula_name=formula)

        selected_mask, pseudo_summary = shot_utility_to_budgeted_summary(
            shot_utility=utility,
            cps=cps_np,
            nfps=nfps_np,
            n_frames=n_frames_int,
            budget_ratio=budget_ratio,
        )

        fscore = vsumm_helper.get_summ_f1score(
            pred_summ=pseudo_summary,
            test_summ=user_summary,
            eval_metric=eval_metric,
        )

        selected_binary = selected_mask.astype(np.float32)
        tau_bin, rho_bin = safe_corr(
            selected_binary,
            shot_gtscore,
            key=f'{key}:{formula}:binary_selection',
        )
        tau_util, rho_util = safe_corr(
            utility,
            shot_gtscore,
            key=f'{key}:{formula}:utility',
        )

        gate_counts = compute_confidence_gate_counts(
            shot_utility=utility,
            selected_mask=selected_mask,
            negative_quantile=negative_quantile,
        )

        summary_len = int(pseudo_summary.sum())
        rows.append({
            'key': str(key),
            'h5_key': h5_key,
            'formula': formula,
            'num_shots': int(cps_np.shape[0]),
            'num_selected_shots': int(selected_mask.sum()),
            'summary_len': summary_len,
            'summary_ratio': float(summary_len / max(n_frames_int, 1)),
            'pseudo_f1': float(fscore),
            'kendall_binary_selection_vs_gtscore': tau_bin,
            'spearman_binary_selection_vs_gtscore': rho_bin,
            'kendall_utility_vs_gtscore': tau_util,
            'spearman_utility_vs_gtscore': rho_util,
            **gate_counts,
        })

    return rows


def summarize_rows(rows: List[Dict], keys: List[str]) -> Dict[str, Dict]:
    key_set = {str(k) for k in keys}
    filtered = [r for r in rows if r['key'] in key_set]
    formulas = sorted(set(r['formula'] for r in filtered))

    out: Dict[str, Dict] = {}
    for formula in formulas:
        sub = [r for r in filtered if r['formula'] == formula]

        def vals(name: str):
            return [float(r[name]) for r in sub if name in r and np.isfinite(float(r[name]))]

        f1s = vals('pseudo_f1')
        taus_bin = vals('kendall_binary_selection_vs_gtscore')
        rhos_bin = vals('spearman_binary_selection_vs_gtscore')
        taus_util = vals('kendall_utility_vs_gtscore')
        rhos_util = vals('spearman_utility_vs_gtscore')
        summary_ratios = vals('summary_ratio')
        pos_rates = vals('positive_shot_rate')
        neg_rates = vals('negative_shot_rate')
        ignore_rates = vals('ignore_shot_rate')

        out[formula] = {
            'num_videos': int(len(sub)),
            'mean_pseudo_f1': safe_nanmean(f1s, default=float('nan')),
            'median_pseudo_f1': float(np.median(f1s)) if f1s else float('nan'),
            'mean_summary_ratio': safe_nanmean(summary_ratios, default=float('nan')),
            'mean_positive_shot_rate': safe_nanmean(pos_rates, default=float('nan')),
            'mean_negative_shot_rate': safe_nanmean(neg_rates, default=float('nan')),
            'mean_ignore_shot_rate': safe_nanmean(ignore_rates, default=float('nan')),
            'mean_kendall_binary_selection_vs_gtscore': safe_nanmean(taus_bin, default=float('nan')),
            'mean_spearman_binary_selection_vs_gtscore': safe_nanmean(rhos_bin, default=float('nan')),
            'mean_kendall_utility_vs_gtscore': safe_nanmean(taus_util, default=float('nan')),
            'mean_spearman_utility_vs_gtscore': safe_nanmean(rhos_util, default=float('nan')),
        }

    return out


def rank_formulas(summary: Dict[str, Dict]) -> List[Dict]:
    ranked = []
    for formula, stats in summary.items():
        f1 = float(stats['mean_pseudo_f1'])
        tau_bin = float(stats['mean_kendall_binary_selection_vs_gtscore'])
        tau_util = float(stats['mean_kendall_utility_vs_gtscore'])
        ignore_rate = float(stats['mean_ignore_shot_rate'])

        # This is diagnostic ranking only. F1 is weighted most because Phase 2B is
        # meant to test budgeted selection teachers, not dense rank teachers.
        score = f1 + 0.10 * tau_bin + 0.05 * tau_util - 0.02 * abs(ignore_rate - 0.65)

        ranked.append({
            'formula': formula,
            'score': score,
            'mean_pseudo_f1': f1,
            'mean_kendall_binary_selection_vs_gtscore': tau_bin,
            'mean_kendall_utility_vs_gtscore': tau_util,
            'mean_positive_shot_rate': float(stats['mean_positive_shot_rate']),
            'mean_negative_shot_rate': float(stats['mean_negative_shot_rate']),
            'mean_ignore_shot_rate': ignore_rate,
            'mean_summary_ratio': float(stats['mean_summary_ratio']),
        })

    ranked.sort(
        key=lambda x: (
            np.nan_to_num(x['score'], nan=-999.0),
            np.nan_to_num(x['mean_pseudo_f1'], nan=-999.0),
            np.nan_to_num(x['mean_kendall_binary_selection_vs_gtscore'], nan=-999.0),
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

    if not (0.0 < args.summary_budget < 1.0):
        raise ValueError(f'Invalid summary budget: {args.summary_budget}')
    if not (0.0 < args.negative_quantile < 1.0):
        raise ValueError(f'Invalid negative quantile: {args.negative_quantile}')

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

    utility_path = (
        Path(args.shot_utility_path)
        if args.shot_utility_path is not None
        else get_dataset_pseudo_dir(args.dataset) / 'shot_utility.npy'
    )
    utility_store = ShotUtilityStore(utility_path)

    all_formula_names = sorted(formula_definitions().keys())
    formulas = args.formulas if args.formulas is not None else all_formula_names
    unknown = sorted(set(formulas) - set(all_formula_names))
    if unknown:
        raise ValueError(f'Unknown formula names: {unknown}. Available: {all_formula_names}')

    out_json = (
        Path(args.out_json)
        if args.out_json is not None
        else Path('diagnostics') / f'{args.dataset}_budgeted_pseudo_summary_teacher.json'
    )
    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else Path('diagnostics') / f'{args.dataset}_budgeted_pseudo_summary_teacher.csv'
    )

    logger.info(
        'Budgeted pseudo-summary teacher | dataset=%s | videos=%d | formulas=%d | '
        'budget=%.3f | negative_q=%.3f | selection_part=%s',
        args.dataset,
        len(selected_keys),
        len(formulas),
        args.summary_budget,
        args.negative_quantile,
        args.selection_part,
    )

    dataset = VideoDatasetMILCond(
        keys=selected_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
    )

    rows: List[Dict] = []
    for idx in range(len(dataset)):
        rows.extend(
            evaluate_one_video_formulas(
                dataset=dataset,
                index=idx,
                utility_store=utility_store,
                formulas=formulas,
                budget_ratio=args.summary_budget,
                negative_quantile=args.negative_quantile,
            )
        )

    summary_all = summarize_rows(rows, selected_keys)
    summary_train = summarize_rows(rows, keys_by_part['train'])
    summary_val = summarize_rows(rows, keys_by_part['val'])
    summary_test = summarize_rows(rows, keys_by_part['test'])
    summary_selection = summarize_rows(rows, recommendation_keys)
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
            'summary_budget': args.summary_budget,
            'negative_quantile': args.negative_quantile,
            'shot_utility_path': str(utility_path),
            'formulas': formulas,
            'num_selected_keys': len(selected_keys),
            'num_recommendation_keys': len(recommendation_keys),
        },
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

    logger.info('Top budgeted pseudo-summary teachers on selection_part=%s:', args.selection_part)
    for item in ranked_selection[:10]:
        logger.info(
            '%s | score=%.4f | pseudo_F1=%.4f | bin_tau=%.4f | util_tau=%.4f | '
            'pos=%.3f | neg=%.3f | ignore=%.3f | len=%.3f',
            item['formula'],
            item['score'],
            item['mean_pseudo_f1'],
            item['mean_kendall_binary_selection_vs_gtscore'],
            item['mean_kendall_utility_vs_gtscore'],
            item['mean_positive_shot_rate'],
            item['mean_negative_shot_rate'],
            item['mean_ignore_shot_rate'],
            item['mean_summary_ratio'],
        )

    logger.info('Wrote JSON: %s', out_json)
    logger.info('Wrote CSV: %s', out_csv)


if __name__ == '__main__':
    main()
