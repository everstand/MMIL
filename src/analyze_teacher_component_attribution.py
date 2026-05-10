#!/usr/bin/env python3
"""Offline oracle/teacher component attribution for SumMe teacher ceiling.

This script reads `gtscore` and `user_summary` only for diagnostics. It does not
train a model, select a checkpoint, or construct pseudo labels for training.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_summe_objective_conflict import (  # noqa: E402
    build_summary_from_bitset,
    exact_oracle_max_user,
    normalize_gtscore,
    rank_or_nan,
    scalar_int,
    summary_to_sample_scores,
)
from helpers import vsumm_helper  # noqa: E402
from helpers.eval_protocol_helper import infer_f1_metric_from_dataset, safe_nanmean  # noqa: E402
from helpers.shot_utility_helper import (  # noqa: E402
    build_budgeted_pseudo_summary_masks,
    build_components,
    compute_formula_utility,
    normalize_01,
)
from run_train_mil_cond import load_all_splits, validate_splits  # noqa: E402


AXIS_COMPONENTS = {
    'motion_action_peak': ['eventiveness', 'visual_change'],
    'boundary_surprise_event_transition': ['caption_change', 'visual_change'],
    'novelty_anti_redundancy': ['distinctiveness', 'anti_redundancy'],
    'semantic_coverage_representativeness': ['semantic', 'representativeness', 'caption_mgs'],
}
UNAVAILABLE_AXES = ['temporal_closure_narrative_completion', 'audio_onset']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline oracle-vs-teacher component attribution for SumMe.'
    )
    parser.add_argument('--dataset', default='summe', choices=('summe',))
    parser.add_argument('--splits', nargs='+', default=['splits/summe.yml'])
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=19500)
    parser.add_argument(
        '--selection-part',
        default='val',
        choices=('all', 'train', 'val', 'test'),
    )
    parser.add_argument('--shot-utility-path', default='pseudo_labels/summe/shot_utility_v3.npy')
    parser.add_argument('--teacher-formula', default='phase1_default')
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--negative-quantile', type=float, default=0.25)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument(
        '--output-json',
        default='diagnostics/summe_teacher_component_attribution_v1_val.json',
    )
    parser.add_argument(
        '--output-csv',
        default='diagnostics/summe_teacher_component_attribution_v1_val_shots.csv',
    )
    return parser.parse_args()


def collect_unique_keys(splits: List[Dict], selection_part: str) -> List[str]:
    keys: List[str] = []
    for split in splits:
        if selection_part == 'all':
            part_keys = split['train_keys'] + split['val_keys'] + split['test_keys']
        elif selection_part == 'train':
            part_keys = split['train_keys']
        elif selection_part == 'val':
            part_keys = split['val_keys']
        elif selection_part == 'test':
            part_keys = split['test_keys']
        else:
            raise ValueError(f'Invalid selection_part={selection_part}')
        keys.extend(part_keys)
    seen = set()
    unique = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def load_shot_utility(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f'Shot utility file not found: {path}')
    obj = np.load(path, allow_pickle=True)
    try:
        obj = obj.item()
    except Exception as exc:
        raise ValueError(f'Invalid npy dict: {path}') from exc
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f'Shot utility file must contain non-empty dict, got {type(obj)}')
    return obj


def open_group(key: str, cache: Dict[str, h5py.File]):
    key_path = Path(key)
    h5_path = str(key_path.parent)
    h5_key = key_path.name
    if h5_path not in cache:
        cache[h5_path] = h5py.File(h5_path, 'r')
    return h5_key, cache[h5_path][h5_key]


def bool_from_keyshot_summary(summary: np.ndarray, cps: np.ndarray) -> np.ndarray:
    summary = np.asarray(summary, dtype=bool).reshape(-1)
    selected = np.zeros((cps.shape[0],), dtype=bool)
    for shot_idx, (first, last) in enumerate(cps):
        selected[shot_idx] = bool(summary[int(first):int(last) + 1].any())
    return selected


def f1_binary(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f'Binary F1 shape mismatch: {pred.shape} vs {target.shape}')
    tp = int((pred & target).sum())
    if tp <= 0:
        return 0.0
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(target.sum()))
    return float(2.0 * precision * recall / (precision + recall))


def precision_recall(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    tp = int((pred & target).sum())
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(target.sum()))
    return float(precision), float(recall)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(f'AUC shape mismatch: {scores.shape} vs {labels.shape}')
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[order[end]] == scores[order[start]]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = float(ranks[labels].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def safe_mean(values: Iterable[float]) -> float:
    return safe_nanmean(list(values), default=float('nan'))


def mean_mask(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape or int(mask.sum()) == 0:
        return float('nan')
    return float(values[mask].mean())


def axis_score(components: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    arrays = [np.asarray(components[name], dtype=np.float32).reshape(-1) for name in names]
    return normalize_01(np.mean(np.stack(arrays, axis=0), axis=0))


def analyze_video(
    key: str,
    group,
    utility_record: Dict,
    teacher_formula: str,
    dataset: str,
    summary_budget: float,
    negative_quantile: float,
) -> Tuple[Dict, List[Dict]]:
    h5_key = str(Path(key).name)
    gtscore = normalize_gtscore(group['gtscore'][...].astype(np.float32))
    user_summary = group['user_summary'][...].astype(np.float32)
    cps = group['change_points'][...].astype(np.int32)
    n_frames = scalar_int(group['n_frames'][...])
    nfps = group['n_frame_per_seg'][...].astype(np.int32).reshape(-1)
    picks = group['picks'][...].astype(np.int32).reshape(-1)
    capacity = int(n_frames * float(summary_budget))
    eval_metric = infer_f1_metric_from_dataset(dataset)

    oracle = exact_oracle_max_user(
        user_summary=user_summary,
        cps=cps,
        nfps=nfps,
        n_frames=n_frames,
        capacity=capacity,
    )
    oracle_summary = build_summary_from_bitset(int(oracle['mask']), cps, n_frames)
    oracle_selected = bool_from_keyshot_summary(oracle_summary, cps)
    oracle_scores = summary_to_sample_scores(oracle_summary, picks, n_frames)
    oracle_tau, oracle_rho = rank_or_nan(oracle_scores, gtscore, key=f'{key}:oracle')

    gtscore_summary = vsumm_helper.get_keyshot_summ(
        gtscore,
        cps,
        n_frames,
        nfps,
        picks,
        proportion=summary_budget,
    )
    gtscore_selected = bool_from_keyshot_summary(gtscore_summary, cps)

    teacher_scores = compute_formula_utility(utility_record, teacher_formula)
    teacher_masks = build_budgeted_pseudo_summary_masks(
        utility=teacher_scores,
        cps=cps,
        nfps=nfps,
        n_frames=n_frames,
        summary_budget=summary_budget,
        negative_quantile=negative_quantile,
    )
    teacher_selected = np.asarray(teacher_masks['selected_mask'], dtype=bool)
    teacher_sample_scores = np.zeros_like(gtscore, dtype=np.float32)
    for sample_idx, frame_idx in enumerate(picks):
        shot_hits = np.where((cps[:, 0] <= int(frame_idx)) & (int(frame_idx) <= cps[:, 1]))[0]
        if shot_hits.size:
            teacher_sample_scores[sample_idx] = float(teacher_scores[int(shot_hits[0])])
    teacher_tau, teacher_rho = rank_or_nan(teacher_sample_scores, gtscore, key=f'{key}:teacher')

    teacher_summary = np.zeros((n_frames,), dtype=bool)
    for shot_idx, is_selected in enumerate(teacher_selected):
        if is_selected:
            first, last = cps[shot_idx]
            teacher_summary[int(first):int(last) + 1] = True
    teacher_f1 = vsumm_helper.get_summ_f1score(
        pred_summ=teacher_summary,
        test_summ=user_summary,
        eval_metric=eval_metric,
    )

    gtscore_f1 = vsumm_helper.get_summ_f1score(
        pred_summ=gtscore_summary,
        test_summ=user_summary,
        eval_metric=eval_metric,
    )
    oracle_f1 = vsumm_helper.get_summ_f1score(
        pred_summ=oracle_summary,
        test_summ=user_summary,
        eval_metric=eval_metric,
    )

    components = build_components(utility_record)
    component_values = {
        name: value for name, value in components.items()
        if isinstance(value, np.ndarray) and value.shape[0] == cps.shape[0]
    }
    for axis_name, names in AXIS_COMPONENTS.items():
        available = [name for name in names if name in component_values]
        if available:
            component_values[f'axis_{axis_name}'] = axis_score(component_values, available)

    t_prec_o, t_rec_o = precision_recall(teacher_selected, oracle_selected)
    t_prec_g, t_rec_g = precision_recall(teacher_selected, gtscore_selected)
    video_record = {
        'key': key,
        'h5_key': h5_key,
        'num_shots': int(cps.shape[0]),
        'n_frames': int(n_frames),
        'oracle_f1': float(oracle_f1),
        'oracle_tau': float(oracle_tau),
        'oracle_rho': float(oracle_rho),
        'gtscore_knapsack_f1': float(gtscore_f1),
        'teacher_f1': float(teacher_f1),
        'teacher_tau': float(teacher_tau),
        'teacher_rho': float(teacher_rho),
        'teacher_vs_oracle_shot_f1': f1_binary(teacher_selected, oracle_selected),
        'teacher_vs_oracle_precision': t_prec_o,
        'teacher_vs_oracle_recall': t_rec_o,
        'teacher_vs_gtscore_shot_f1': f1_binary(teacher_selected, gtscore_selected),
        'teacher_vs_gtscore_precision': t_prec_g,
        'teacher_vs_gtscore_recall': t_rec_g,
        'oracle_selected_count': int(oracle_selected.sum()),
        'gtscore_selected_count': int(gtscore_selected.sum()),
        'teacher_selected_count': int(teacher_selected.sum()),
        'oracle_missed_by_teacher_count': int((oracle_selected & ~teacher_selected).sum()),
        'teacher_extra_vs_oracle_count': int((teacher_selected & ~oracle_selected).sum()),
    }

    shot_rows: List[Dict] = []
    for shot_idx in range(cps.shape[0]):
        row = {
            'key': key,
            'h5_key': h5_key,
            'shot_idx': int(shot_idx),
            'start_frame': int(cps[shot_idx, 0]),
            'end_frame': int(cps[shot_idx, 1]),
            'nfps': int(nfps[shot_idx]),
            'oracle_selected': int(oracle_selected[shot_idx]),
            'gtscore_knapsack_selected': int(gtscore_selected[shot_idx]),
            'teacher_selected': int(teacher_selected[shot_idx]),
            'teacher_score': float(teacher_scores[shot_idx]),
        }
        for comp_name, values in component_values.items():
            row[comp_name] = float(values[shot_idx])
        shot_rows.append(row)

    return video_record, shot_rows


def summarize(video_records: List[Dict], shot_rows: List[Dict]) -> Dict:
    summary = {
        'num_videos': int(len(video_records)),
        'num_shots': int(len(shot_rows)),
    }
    for key in [
        'oracle_f1', 'oracle_tau', 'oracle_rho',
        'gtscore_knapsack_f1', 'teacher_f1', 'teacher_tau', 'teacher_rho',
        'teacher_vs_oracle_shot_f1', 'teacher_vs_oracle_precision', 'teacher_vs_oracle_recall',
        'teacher_vs_gtscore_shot_f1', 'teacher_vs_gtscore_precision', 'teacher_vs_gtscore_recall',
        'oracle_missed_by_teacher_count', 'teacher_extra_vs_oracle_count',
    ]:
        summary[f'mean_{key}'] = safe_mean(record[key] for record in video_records)

    if not shot_rows:
        return summary

    component_names = [
        key for key in shot_rows[0].keys()
        if key not in {
            'key', 'h5_key', 'shot_idx', 'start_frame', 'end_frame', 'nfps',
            'oracle_selected', 'gtscore_knapsack_selected', 'teacher_selected',
        }
    ]
    oracle = np.asarray([row['oracle_selected'] for row in shot_rows], dtype=bool)
    gtscore = np.asarray([row['gtscore_knapsack_selected'] for row in shot_rows], dtype=bool)
    teacher = np.asarray([row['teacher_selected'] for row in shot_rows], dtype=bool)
    missed = oracle & ~teacher
    extra = teacher & ~oracle

    component_stats = {}
    for name in component_names:
        values = np.asarray([row[name] for row in shot_rows], dtype=np.float32)
        component_stats[name] = {
            'mean_oracle_selected': mean_mask(values, oracle),
            'mean_gtscore_selected': mean_mask(values, gtscore),
            'mean_teacher_selected': mean_mask(values, teacher),
            'mean_oracle_missed_by_teacher': mean_mask(values, missed),
            'mean_teacher_extra_vs_oracle': mean_mask(values, extra),
            'auc_oracle_selected': rank_auc(values, oracle),
            'auc_gtscore_selected': rank_auc(values, gtscore),
            'auc_teacher_selected': rank_auc(values, teacher),
            'missed_minus_teacher_selected': mean_mask(values, missed) - mean_mask(values, teacher),
            'oracle_minus_teacher_selected': mean_mask(values, oracle) - mean_mask(values, teacher),
        }

    axis_summary = {}
    for axis_name, names in AXIS_COMPONENTS.items():
        axis_key = f'axis_{axis_name}'
        axis_summary[axis_name] = component_stats.get(axis_key, {'available': False})
        axis_summary[axis_name]['components'] = names
        axis_summary[axis_name]['available'] = axis_key in component_stats
    for axis_name in UNAVAILABLE_AXES:
        axis_summary[axis_name] = {
            'available': False,
            'reason': 'No corresponding non-human component is present in shot_utility_v3.',
        }

    summary['component_stats'] = component_stats
    summary['axis_summary'] = axis_summary
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    splits = load_all_splits(args.splits, val_ratio=args.val_ratio, seed=args.seed)
    validate_splits(splits, expected_dataset=args.dataset)
    keys = collect_unique_keys(splits, args.selection_part)
    if args.max_videos is not None:
        keys = keys[:args.max_videos]
    if not keys:
        raise ValueError('No selected keys.')

    utilities = load_shot_utility(Path(args.shot_utility_path))
    h5_cache: Dict[str, h5py.File] = {}
    video_records: List[Dict] = []
    shot_rows: List[Dict] = []
    try:
        for key in keys:
            h5_key, group = open_group(key, h5_cache)
            if h5_key not in utilities:
                raise KeyError(f'Missing {h5_key} in {args.shot_utility_path}')
            video_record, rows = analyze_video(
                key=key,
                group=group,
                utility_record=utilities[h5_key],
                teacher_formula=args.teacher_formula,
                dataset=args.dataset,
                summary_budget=args.summary_budget,
                negative_quantile=args.negative_quantile,
            )
            video_records.append(video_record)
            shot_rows.extend(rows)
    finally:
        for handle in h5_cache.values():
            handle.close()

    summary = summarize(video_records, shot_rows)
    output = {
        'meta': {
            'dataset': args.dataset,
            'splits': args.splits,
            'selection_part': args.selection_part,
            'seed': int(args.seed),
            'val_ratio': float(args.val_ratio),
            'shot_utility_path': args.shot_utility_path,
            'teacher_formula': args.teacher_formula,
            'summary_budget': float(args.summary_budget),
            'diagnostic_only': True,
            'uses_human_labels_for_training': False,
            'human_label_use': 'offline oracle/component attribution only',
        },
        'summary': summary,
        'videos': video_records,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    write_csv(Path(args.output_csv), shot_rows)

    printable = {
        'num_videos': summary['num_videos'],
        'num_shots': summary['num_shots'],
        'mean_teacher_f1': summary['mean_teacher_f1'],
        'mean_teacher_tau': summary['mean_teacher_tau'],
        'mean_teacher_rho': summary['mean_teacher_rho'],
        'mean_teacher_vs_oracle_shot_f1': summary['mean_teacher_vs_oracle_shot_f1'],
        'mean_teacher_vs_oracle_recall': summary['mean_teacher_vs_oracle_recall'],
        'axis_summary': summary['axis_summary'],
        'output_json': str(output_json),
        'output_csv': args.output_csv,
    }
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()