#!/usr/bin/env python3
"""Build/evaluate one budgeted shot-value teacher candidate.

The candidate is non-human: it uses only existing shot_utility_v3 components.
Human `gtscore` and `user_summary` are used only for teacher-alone validation
and gate reporting. No student training is run here.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_summe_objective_conflict import normalize_gtscore, rank_or_nan, scalar_int  # noqa: E402
from analyze_teacher_component_attribution import (  # noqa: E402
    bool_from_keyshot_summary,
    collect_unique_keys,
    load_shot_utility,
    open_group,
)
from helpers import vsumm_helper  # noqa: E402
from helpers.eval_protocol_helper import infer_f1_metric_from_dataset, safe_nanmean  # noqa: E402
from helpers.shot_utility_helper import build_components, normalize_01  # noqa: E402
from run_train_mil_cond import load_all_splits, validate_splits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Teacher-alone budgeted shot-value diagnostic.')
    parser.add_argument('--dataset', default='summe', choices=('summe',))
    parser.add_argument('--splits', nargs='+', default=['splits/summe.yml'])
    parser.add_argument('--selection-part', default='val', choices=('all', 'train', 'val', 'test'))
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=19500)
    parser.add_argument('--shot-utility-path', default='pseudo_labels/summe/shot_utility_v3.npy')
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--length-alpha', type=float, default=0.25)
    parser.add_argument('--length-min', type=float, default=0.70)
    parser.add_argument('--length-max', type=float, default=1.30)
    parser.add_argument('--gate-f1', type=float, default=0.55)
    parser.add_argument('--gate-tau', type=float, default=0.20)
    parser.add_argument('--gate-rho', type=float, default=0.25)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument('--output-json', default='diagnostics/summe_budgeted_shot_value_teacher_v1_val.json')
    parser.add_argument('--output-csv', default='diagnostics/summe_budgeted_shot_value_teacher_v1_val.csv')
    parser.add_argument('--teacher-output', default='')
    parser.add_argument('--force-write', action='store_true')
    return parser.parse_args()


def budgeted_shot_value_scores(record: Dict, nfps: np.ndarray, args) -> Dict[str, np.ndarray]:
    components = build_components(record)
    num_shots = int(np.asarray(nfps).reshape(-1).shape[0])
    for name in ('semantic', 'representativeness', 'caption_mgs', 'anti_redundancy', 'eventiveness', 'visual_change', 'caption_change'):
        value = components.get(name)
        if not isinstance(value, np.ndarray) or value.shape[0] != num_shots:
            raise ValueError(f'Missing or invalid component for budgeted value: {name}')

    # Attribution showed local peak/novelty axes are over-selected by the old teacher.
    # v1 therefore treats them as weak modifiers and keeps coverage/representativeness as the base value.
    coverage_value = normalize_01(
        0.50 * components['semantic']
        + 0.30 * components['representativeness']
        + 0.20 * components['caption_mgs']
    )
    weak_context = normalize_01(
        0.35 * components['anti_redundancy']
        + 0.25 * components['eventiveness']
        + 0.25 * components['visual_change']
        + 0.15 * components['caption_change']
    )
    base_value = normalize_01(0.85 * coverage_value + 0.15 * weak_context)

    nfps = np.asarray(nfps, dtype=np.float32).reshape(-1)
    median_len = float(np.median(nfps[nfps > 0])) if np.any(nfps > 0) else 1.0
    length_factor = np.power(np.clip(nfps / max(median_len, 1.0), 1e-6, None), float(args.length_alpha))
    length_factor = np.clip(length_factor, float(args.length_min), float(args.length_max))
    shot_value = normalize_01(base_value * length_factor)

    return {
        'shot_scores': shot_value.astype(np.float32),
        'coverage_value': coverage_value.astype(np.float32),
        'weak_context': weak_context.astype(np.float32),
        'length_factor': length_factor.astype(np.float32),
    }


def project_shot_scores_to_sampled(shot_scores: np.ndarray, cps: np.ndarray, picks: np.ndarray) -> np.ndarray:
    shot_scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    cps = np.asarray(cps, dtype=np.int32)
    picks = np.asarray(picks, dtype=np.int32).reshape(-1)
    out = np.zeros((picks.shape[0],), dtype=np.float32)
    for idx, frame_idx in enumerate(picks):
        hits = np.where((cps[:, 0] <= int(frame_idx)) & (int(frame_idx) <= cps[:, 1]))[0]
        if hits.size:
            out[idx] = float(shot_scores[int(hits[0])])
    return out


def selected_summary_from_shot_mask(selected: np.ndarray, cps: np.ndarray, n_frames: int) -> np.ndarray:
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    summary = np.zeros((int(n_frames),), dtype=bool)
    for shot_idx, is_selected in enumerate(selected):
        if is_selected:
            first, last = cps[shot_idx]
            summary[int(first):int(last) + 1] = True
    return summary


def evaluate_video(key: str, group, utility_record: Dict, args) -> Dict:
    h5_key = str(Path(key).name)
    gtscore = normalize_gtscore(group['gtscore'][...].astype(np.float32))
    user_summary = group['user_summary'][...].astype(np.float32)
    cps = group['change_points'][...].astype(np.int32)
    n_frames = scalar_int(group['n_frames'][...])
    nfps = group['n_frame_per_seg'][...].astype(np.int32).reshape(-1)
    picks = group['picks'][...].astype(np.int32).reshape(-1)
    eval_metric = infer_f1_metric_from_dataset(args.dataset)

    scores = budgeted_shot_value_scores(utility_record, nfps, args)
    shot_scores = scores['shot_scores']
    values = np.round(shot_scores * 1000.0).astype(np.int32)
    capacity = int(n_frames * float(args.summary_budget))
    if values.size == 0 or int(values.max()) <= 0:
        selected_idx: List[int] = []
    else:
        selected_idx = vsumm_helper.knapsack(values.tolist(), nfps.tolist(), capacity)
    selected = np.zeros((cps.shape[0],), dtype=bool)
    selected[selected_idx] = True
    pred_summary = selected_summary_from_shot_mask(selected, cps, n_frames)
    f1 = vsumm_helper.get_summ_f1score(
        pred_summ=pred_summary,
        test_summ=user_summary,
        eval_metric=eval_metric,
    )
    sampled_scores = project_shot_scores_to_sampled(shot_scores, cps, picks)
    tau, rho = rank_or_nan(sampled_scores, gtscore, key=f'{key}:budgeted_shot_value_v1')

    gtscore_summary = vsumm_helper.get_keyshot_summ(
        gtscore,
        cps,
        n_frames,
        nfps,
        picks,
        proportion=float(args.summary_budget),
    )
    gtscore_selected = bool_from_keyshot_summary(gtscore_summary, cps)
    return {
        'key': key,
        'h5_key': h5_key,
        'num_shots': int(cps.shape[0]),
        'n_frames': int(n_frames),
        'selected_count': int(selected.sum()),
        'budget_ratio': float(pred_summary.sum() / max(1, n_frames)),
        'f1': float(f1),
        'kendall': float(tau),
        'spearman': float(rho),
        'selected_vs_gtscore_shot_f1': float(f1_shot(selected, gtscore_selected)),
        'mean_selected_score': float(shot_scores[selected].mean()) if selected.any() else float('nan'),
        'mean_unselected_score': float(shot_scores[~selected].mean()) if (~selected).any() else float('nan'),
    }


def f1_shot(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    tp = int((pred & target).sum())
    if tp <= 0:
        return 0.0
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(target.sum()))
    return float(2.0 * precision * recall / (precision + recall))


def summarize(records: List[Dict], args) -> Dict:
    def vals(name: str) -> List[float]:
        return [float(record[name]) for record in records]

    summary = {
        'num_videos': int(len(records)),
        'mean_f1': safe_nanmean(vals('f1'), default=float('nan')),
        'mean_kendall': safe_nanmean(vals('kendall'), default=float('nan')),
        'mean_spearman': safe_nanmean(vals('spearman'), default=float('nan')),
        'mean_budget_ratio': safe_nanmean(vals('budget_ratio'), default=float('nan')),
        'mean_selected_vs_gtscore_shot_f1': safe_nanmean(vals('selected_vs_gtscore_shot_f1'), default=float('nan')),
        'gate_f1': float(args.gate_f1),
        'gate_tau': float(args.gate_tau),
        'gate_rho': float(args.gate_rho),
    }
    summary['train_ready'] = bool(
        summary['mean_f1'] >= float(args.gate_f1)
        and summary['mean_kendall'] > float(args.gate_tau)
        and summary['mean_spearman'] > float(args.gate_rho)
    )
    return summary


def write_csv(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def maybe_write_teacher(path: str, utilities: Dict[str, Dict], selected_keys: List[str], args, train_ready: bool) -> None:
    if not path:
        return
    if not train_ready and not args.force_write:
        print(f'SKIP_TEACHER_WRITE gate_failed path={path}')
        return
    out = {}
    for key in selected_keys:
        h5_key = Path(key).name
        # H5 nfps is required for length-aware score; load it directly here.
        with h5py.File(str(Path(key).parent), 'r') as h5:
            nfps = h5[h5_key]['n_frame_per_seg'][...].astype(np.int32).reshape(-1)
        scores = budgeted_shot_value_scores(utilities[h5_key], nfps, args)
        out[h5_key] = {
            'shot_scores': scores['shot_scores'],
            'coverage_value': scores['coverage_value'],
            'weak_context': scores['weak_context'],
            'length_factor': scores['length_factor'],
            'meta': {
                'teacher_type': 'budgeted_shot_value_v1',
                'uses_human_training_labels': False,
                'length_alpha': float(args.length_alpha),
                'length_min': float(args.length_min),
                'length_max': float(args.length_max),
            },
        }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    print(f'WROTE_TEACHER {output_path}')


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
    records: List[Dict] = []
    try:
        for key in keys:
            h5_key, group = open_group(key, h5_cache)
            if h5_key not in utilities:
                raise KeyError(f'Missing {h5_key} in {args.shot_utility_path}')
            records.append(evaluate_video(key, group, utilities[h5_key], args))
    finally:
        for handle in h5_cache.values():
            handle.close()

    summary = summarize(records, args)
    output = {
        'meta': {
            'dataset': args.dataset,
            'splits': args.splits,
            'selection_part': args.selection_part,
            'seed': int(args.seed),
            'val_ratio': float(args.val_ratio),
            'summary_budget': float(args.summary_budget),
            'shot_utility_path': args.shot_utility_path,
            'teacher_type': 'budgeted_shot_value_v1',
            'uses_human_labels_for_training': False,
            'human_label_use': 'teacher-alone validation gate only',
            'formula': '0.85*(0.50 semantic + 0.30 representativeness + 0.20 caption_mgs) + 0.15*(anti_redundancy/eventiveness/visual_change/caption_change), length-adjusted',
            'length_alpha': float(args.length_alpha),
        },
        'summary': summary,
        'records': records,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    write_csv(Path(args.output_csv), records)
    maybe_write_teacher(args.teacher_output, utilities, keys, args, summary['train_ready'])
    print(json.dumps({'summary': summary, 'output_json': str(output_json), 'output_csv': args.output_csv}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()