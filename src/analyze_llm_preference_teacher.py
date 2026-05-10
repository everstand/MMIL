# -*- coding: utf-8 -*-
"""Validation-only diagnostics for LLM preference teachers."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from helpers import vsumm_helper
from helpers.eval_protocol_helper import (
    compute_rank_metrics_from_gtscore,
    infer_f1_metric_from_key,
    safe_nanmean,
)
from helpers.preference_teacher_helper import PreferenceTeacherStore, normalize_01
from helpers.shot_utility_helper import ShotUtilityStore
from run_train_mil_cond import load_all_splits, validate_splits


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=('summe', 'tvsum'))
    parser.add_argument('--splits', type=str, nargs='+', required=True)
    parser.add_argument('--teacher-path', type=str, required=True)
    parser.add_argument('--baseline-shot-utility-path', type=str, default=None)
    parser.add_argument('--baseline-formula', type=str, default='phase1_default')
    parser.add_argument('--selection-part', type=str, default='val', choices=('train', 'val', 'test', 'all'))
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=(
            'Seed used by load_all_splits for train/val partitioning. '
            'Defaults to the canonical training seed for the dataset: '
            '19500 for SumMe, 12345 for TVSum.'
        ),
    )
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--output-json', type=str, default=None)
    parser.add_argument('--output-csv', type=str, default=None)
    parser.add_argument('--min-f1', type=float, default=0.45)
    parser.add_argument('--min-tau', type=float, default=0.08)
    parser.add_argument('--min-rho', type=float, default=0.10)
    return parser


def shot_scores_to_sampled_scores(shot_scores: np.ndarray,
                                  cps: np.ndarray,
                                  picks: np.ndarray,
                                  key: str) -> np.ndarray:
    scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    cps = np.asarray(cps, dtype=np.int64)
    picks = np.asarray(picks, dtype=np.int64).reshape(-1)
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Invalid change_points shape for {key}: {cps.shape}')
    if scores.shape[0] != cps.shape[0]:
        raise ValueError(f'shot score length mismatch for {key}: {scores.shape[0]} vs {cps.shape[0]}')
    sampled = np.zeros((picks.shape[0],), dtype=np.float32)
    assigned = np.zeros((picks.shape[0],), dtype=bool)
    for shot_idx, (first, last) in enumerate(cps.tolist()):
        mask = (picks >= int(first)) & (picks <= int(last))
        sampled[mask] = float(scores[shot_idx])
        assigned[mask] = True
    if not assigned.all():
        missing = np.where(~assigned)[0]
        for idx in missing.tolist():
            frame = int(picks[idx])
            nearest = int(np.argmin(np.minimum(np.abs(cps[:, 0] - frame), np.abs(cps[:, 1] - frame))))
            sampled[idx] = float(scores[nearest])
    return sampled.astype(np.float32)


def evaluate_sample_scores(sample_scores: np.ndarray,
                           group,
                           key: str) -> Dict[str, float]:
    cps = group['change_points'][...].astype(np.int32)
    nfps = group['n_frame_per_seg'][...].astype(np.int32)
    n_frames = int(np.asarray(group['n_frames'][...]).item())
    picks = group['picks'][...].astype(np.int32)
    pred_summ = vsumm_helper.get_keyshot_summ(
        sample_scores,
        cps,
        n_frames,
        nfps,
        picks,
    )
    fscore = vsumm_helper.get_summ_f1score(
        pred_summ=pred_summ,
        test_summ=group['user_summary'][...].astype(np.float32),
        eval_metric=infer_f1_metric_from_key(key),
    )
    rank_metrics = compute_rank_metrics_from_gtscore(
        pred_scores=sample_scores,
        gtscore=group['gtscore'][...].astype(np.float32),
        key=key,
    )
    return {
        'F1': float(fscore),
        'Tau': float(rank_metrics['kendall']),
        'Rho': float(rank_metrics['spearman']),
        'budget_ratio': float(np.asarray(pred_summ, dtype=np.float32).sum() / max(float(n_frames), 1.0)),
    }


def selected_keys_for_split(split: Dict, selection_part: str) -> List[str]:
    if selection_part == 'train':
        return list(split['train_keys'])
    if selection_part == 'val':
        return list(split['val_keys'])
    if selection_part == 'test':
        return list(split['test_keys'])
    return list(split['train_keys']) + list(split['val_keys']) + list(split['test_keys'])


def summarize(rows: List[Dict]) -> Dict[str, float]:
    out = {
        'num_rows': int(len(rows)),
        'teacher_F1': float(np.mean([row['teacher_F1'] for row in rows])) if rows else 0.0,
        'teacher_Tau': safe_nanmean([row['teacher_Tau'] for row in rows]),
        'teacher_Rho': safe_nanmean([row['teacher_Rho'] for row in rows]),
        'teacher_budget_ratio': float(np.mean([row['teacher_budget_ratio'] for row in rows])) if rows else 0.0,
        'num_positive_shots': float(np.mean([row['num_positive_shots'] for row in rows])) if rows else 0.0,
        'num_negative_shots': float(np.mean([row['num_negative_shots'] for row in rows])) if rows else 0.0,
        'agreement_rate': float(np.mean([row['agreement_rate'] for row in rows])) if rows else 0.0,
    }
    if rows and 'baseline_F1' in rows[0]:
        out.update({
            'baseline_F1': float(np.mean([row['baseline_F1'] for row in rows])),
            'baseline_Tau': safe_nanmean([row['baseline_Tau'] for row in rows]),
            'baseline_Rho': safe_nanmean([row['baseline_Rho'] for row in rows]),
            'baseline_budget_ratio': float(np.mean([row['baseline_budget_ratio'] for row in rows])),
        })
        out.update({
            'delta_F1': out['teacher_F1'] - out['baseline_F1'],
            'delta_Tau': out['teacher_Tau'] - out['baseline_Tau'],
            'delta_Rho': out['teacher_Rho'] - out['baseline_Rho'],
        })
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('')
        return
    fieldnames = sorted({name for row in rows for name in row.keys()})
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = get_parser().parse_args()
    effective_seed = int(args.seed) if args.seed is not None else default_split_seed(args.dataset)
    splits = load_all_splits(args.splits, val_ratio=args.val_ratio, seed=effective_seed)
    validate_splits(splits, args.dataset)
    teacher_store = PreferenceTeacherStore(Path(args.teacher_path))
    baseline_store = None
    if args.baseline_shot_utility_path:
        baseline_store = ShotUtilityStore(Path(args.baseline_shot_utility_path))

    rows: List[Dict] = []
    for fold_idx, split in enumerate(splits):
        for key in selected_keys_for_split(split, args.selection_part):
            key_path = Path(key)
            h5_path = key_path.parent
            h5_key = key_path.name
            with h5py.File(str(h5_path), 'r') as h5:
                group = h5[h5_key]
                cps = group['change_points'][...].astype(np.int32)
                picks = group['picks'][...].astype(np.int32)
                teacher_record = teacher_store.get(h5_key)
                teacher_shot_scores = np.asarray(teacher_record['shot_scores'], dtype=np.float32)
                teacher_sample_scores = shot_scores_to_sampled_scores(
                    teacher_shot_scores,
                    cps=cps,
                    picks=picks,
                    key=str(key),
                )
                teacher_metrics = evaluate_sample_scores(teacher_sample_scores, group, key=str(key))
                inclusion_prob = np.asarray(teacher_record['inclusion_prob'], dtype=np.float32)
                confidence = np.asarray(teacher_record['teacher_confidence'], dtype=np.float32)
                row = {
                    'fold_idx': int(fold_idx),
                    'h5_key': h5_key,
                    'teacher_F1': teacher_metrics['F1'],
                    'teacher_Tau': teacher_metrics['Tau'],
                    'teacher_Rho': teacher_metrics['Rho'],
                    'teacher_budget_ratio': teacher_metrics['budget_ratio'],
                    'num_positive_shots': int((inclusion_prob >= 0.60).sum()),
                    'num_negative_shots': int((inclusion_prob <= 0.20).sum()),
                    'num_pairs': int(np.asarray(teacher_record['pair_i']).reshape(-1).shape[0]),
                    'agreement_rate': float(confidence.mean()) if confidence.size else 0.0,
                }
                if baseline_store is not None:
                    baseline_shot_scores = normalize_01(
                        baseline_store.get(h5_key=h5_key, formula_name=args.baseline_formula)
                    )
                    baseline_sample_scores = shot_scores_to_sampled_scores(
                        baseline_shot_scores,
                        cps=cps,
                        picks=picks,
                        key=str(key),
                    )
                    baseline_metrics = evaluate_sample_scores(baseline_sample_scores, group, key=str(key))
                    row.update({
                        'baseline_F1': baseline_metrics['F1'],
                        'baseline_Tau': baseline_metrics['Tau'],
                        'baseline_Rho': baseline_metrics['Rho'],
                        'baseline_budget_ratio': baseline_metrics['budget_ratio'],
                        'delta_F1': teacher_metrics['F1'] - baseline_metrics['F1'],
                        'delta_Tau': teacher_metrics['Tau'] - baseline_metrics['Tau'],
                        'delta_Rho': teacher_metrics['Rho'] - baseline_metrics['Rho'],
                    })
                rows.append(row)

    summary = summarize(rows)
    absolute_pass = (
        summary['teacher_F1'] > float(args.min_f1)
        and summary['teacher_Tau'] > float(args.min_tau)
        and summary['teacher_Rho'] > float(args.min_rho)
    )
    delta_pass_count = 0
    if 'baseline_F1' in summary:
        delta_pass_count = int(summary['delta_F1'] > 0.0) + int(summary['delta_Tau'] > 0.0) + int(summary['delta_Rho'] > 0.0)
    else:
        delta_pass_count = 3
    train_ready = bool(absolute_pass and delta_pass_count >= 2)

    result = {
        'dataset': args.dataset,
        'selection_part': args.selection_part,
        'seed': int(effective_seed),
        'seed_explicit': bool(args.seed is not None),
        'val_ratio': float(args.val_ratio),
        'splits': list(args.splits),
        'teacher_path': args.teacher_path,
        'baseline_shot_utility_path': args.baseline_shot_utility_path,
        'baseline_formula': args.baseline_formula,
        'thresholds': {
            'min_F1': float(args.min_f1),
            'min_Tau': float(args.min_tau),
            'min_Rho': float(args.min_rho),
            'min_delta_metrics': 2,
        },
        'summary': summary,
        'absolute_pass': bool(absolute_pass),
        'delta_pass_count': int(delta_pass_count),
        'train_ready': train_ready,
        'rows': rows,
    }

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, sort_keys=True)
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)

    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2, sort_keys=True))
    if not train_ready:
        raise SystemExit(2)


def default_split_seed(dataset: str) -> int:
    if dataset == 'summe':
        return 19500
    if dataset == 'tvsum':
        return 12345
    raise ValueError(f'Unsupported dataset: {dataset}')


if __name__ == '__main__':
    main()
