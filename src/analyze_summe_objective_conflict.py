#!/usr/bin/env python3
"""Diagnose SumMe objective conflict between F1 keyshot overlap and gtscore rank.

This script is offline-only. It reads `gtscore` and `user_summary` only for
benchmark/protocol diagnostics. It does not train a model, choose a checkpoint,
or construct any pseudo label used by training.
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

from helpers import vsumm_helper
from helpers.eval_protocol_helper import (  # noqa: E402
    compute_rank_metrics_from_gtscore,
    infer_f1_metric_from_dataset,
    safe_nanmean,
)
from run_train_mil_cond import load_all_splits, validate_splits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline SumMe diagnostic for F1-vs-rank objective conflict.'
    )
    parser.add_argument('--dataset', default='summe', choices=('summe',))
    parser.add_argument('--splits', nargs='+', default=['splits/summe.yml'])
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=19500)
    parser.add_argument(
        '--selection-part',
        default='all',
        choices=('all', 'train', 'val', 'test'),
        help='Unique videos to diagnose after loading splits with the requested seed.',
    )
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument(
        '--output-json',
        default='diagnostics/summe_objective_conflict_v1.json',
    )
    parser.add_argument(
        '--output-csv',
        default='diagnostics/summe_objective_conflict_v1.csv',
    )
    return parser.parse_args()


def normalize_gtscore(gtscore: np.ndarray) -> np.ndarray:
    gtscore = np.asarray(gtscore, dtype=np.float32).reshape(-1)
    if gtscore.size == 0:
        raise ValueError('Empty gtscore.')
    gtscore = gtscore - float(gtscore.min())
    max_value = float(gtscore.max())
    if max_value > 0.0:
        gtscore = gtscore / max_value
    return gtscore.astype(np.float32)


def scalar_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


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
            unique.append(key)
            seen.add(key)
    return unique


def summary_to_sample_scores(summary: np.ndarray, picks: np.ndarray, n_frames: int) -> np.ndarray:
    summary = np.asarray(summary, dtype=np.float32).reshape(-1)
    picks = np.asarray(picks, dtype=np.int32).reshape(-1)
    if summary.shape[0] != int(n_frames):
        raise ValueError(f'Summary/n_frames mismatch: {summary.shape[0]} vs {n_frames}')
    scores = np.zeros((picks.shape[0],), dtype=np.float32)
    for idx, start in enumerate(picks):
        end = int(picks[idx + 1]) if idx + 1 < picks.shape[0] else int(n_frames)
        start = int(start)
        if end <= start:
            scores[idx] = float(summary[start])
        else:
            scores[idx] = float(summary[start:end].mean())
    return scores


def rank_or_nan(pred_scores: np.ndarray, gtscore: np.ndarray, key: str) -> Tuple[float, float]:
    try:
        metrics = compute_rank_metrics_from_gtscore(
            pred_scores=np.asarray(pred_scores, dtype=np.float32),
            gtscore=np.asarray(gtscore, dtype=np.float32),
            key=key,
        )
        return float(metrics['kendall']), float(metrics['spearman'])
    except Exception:
        return float('nan'), float('nan')


def popcount(mask: int) -> int:
    return bin(int(mask)).count('1')


def bitset_to_indices(mask: int, num_shots: int) -> List[int]:
    return [idx for idx in range(num_shots) if (int(mask) >> idx) & 1]


def build_summary_from_bitset(mask: int, cps: np.ndarray, n_frames: int) -> np.ndarray:
    summary = np.zeros((int(n_frames),), dtype=bool)
    for shot_idx in bitset_to_indices(mask, int(cps.shape[0])):
        first, last = cps[shot_idx]
        summary[int(first):int(last) + 1] = True
    return summary


def exact_oracle_for_user(
    user_summary: np.ndarray,
    cps: np.ndarray,
    nfps: np.ndarray,
    n_frames: int,
    capacity: int,
) -> Dict:
    user_summary = np.asarray(user_summary, dtype=bool).reshape(-1)
    if user_summary.shape[0] != int(n_frames):
        raise ValueError(
            f'user_summary/n_frames mismatch: {user_summary.shape[0]} vs {n_frames}'
        )

    cps = np.asarray(cps, dtype=np.int32)
    nfps = np.asarray(nfps, dtype=np.int32).reshape(-1)
    shot_lengths = (cps[:, 1] - cps[:, 0] + 1).astype(np.int32)
    overlaps = np.zeros((cps.shape[0],), dtype=np.int32)
    for shot_idx, (first, last) in enumerate(cps):
        overlaps[shot_idx] = int(user_summary[int(first):int(last) + 1].sum())

    gt_len = int(user_summary.sum())
    if gt_len <= 0:
        return {
            'f1': 0.0,
            'mask': 0,
            'overlap': 0,
            'budget_weight': 0,
            'selected_frames': 0,
            'gt_frames': gt_len,
        }

    if np.array_equal(shot_lengths, nfps):
        return exact_oracle_1d(overlaps, nfps, gt_len, capacity)
    return exact_oracle_2d(overlaps, nfps, shot_lengths, gt_len, capacity)


def better_state(candidate, incumbent, gt_len: int) -> bool:
    cand_overlap, cand_weight, cand_mask = candidate
    inc_overlap, inc_weight, inc_mask = incumbent
    cand_f1 = 0.0 if cand_weight <= 0 else 2.0 * cand_overlap / (cand_weight + gt_len)
    inc_f1 = 0.0 if inc_weight <= 0 else 2.0 * inc_overlap / (inc_weight + gt_len)
    if cand_f1 > inc_f1 + 1e-12:
        return True
    if abs(cand_f1 - inc_f1) <= 1e-12:
        if cand_overlap != inc_overlap:
            return cand_overlap > inc_overlap
        if cand_weight != inc_weight:
            return cand_weight < inc_weight
        return popcount(cand_mask) < popcount(inc_mask)
    return False


def exact_oracle_1d(overlaps: np.ndarray, weights: np.ndarray, gt_len: int, capacity: int) -> Dict:
    best_overlap = np.full((capacity + 1,), -1, dtype=np.int32)
    best_mask: List[int] = [0 for _ in range(capacity + 1)]
    best_overlap[0] = 0

    for shot_idx, (weight, overlap) in enumerate(zip(weights.tolist(), overlaps.tolist())):
        weight = int(weight)
        overlap = int(overlap)
        if weight <= 0:
            continue
        for budget in range(capacity, weight - 1, -1):
            prev_overlap = int(best_overlap[budget - weight])
            if prev_overlap < 0:
                continue
            cand_overlap = prev_overlap + overlap
            cand_mask = best_mask[budget - weight] | (1 << shot_idx)
            if cand_overlap > int(best_overlap[budget]):
                best_overlap[budget] = cand_overlap
                best_mask[budget] = cand_mask
            elif cand_overlap == int(best_overlap[budget]):
                if popcount(cand_mask) < popcount(best_mask[budget]):
                    best_mask[budget] = cand_mask

    best = (0, 0, 0)
    for budget in range(capacity + 1):
        overlap = int(best_overlap[budget])
        if overlap < 0:
            continue
        candidate = (overlap, int(budget), int(best_mask[budget]))
        if better_state(candidate, best, gt_len):
            best = candidate

    overlap, selected_frames, mask = best
    f1 = 0.0 if selected_frames <= 0 else 2.0 * overlap / (selected_frames + gt_len)
    return {
        'f1': float(f1),
        'mask': int(mask),
        'overlap': int(overlap),
        'budget_weight': int(selected_frames),
        'selected_frames': int(selected_frames),
        'gt_frames': int(gt_len),
    }


def exact_oracle_2d(
    overlaps: np.ndarray,
    budget_weights: np.ndarray,
    shot_lengths: np.ndarray,
    gt_len: int,
    capacity: int,
) -> Dict:
    states: Dict[Tuple[int, int], Tuple[int, int]] = {(0, 0): (0, 0)}
    for shot_idx, (budget_weight, shot_len, overlap) in enumerate(
        zip(budget_weights.tolist(), shot_lengths.tolist(), overlaps.tolist())
    ):
        budget_weight = int(budget_weight)
        shot_len = int(shot_len)
        overlap = int(overlap)
        if budget_weight <= 0 or shot_len <= 0:
            continue
        updates = dict(states)
        for (budget, selected_len), (prev_overlap, prev_mask) in states.items():
            next_budget = budget + budget_weight
            if next_budget > capacity:
                continue
            next_len = selected_len + shot_len
            next_overlap = prev_overlap + overlap
            next_mask = prev_mask | (1 << shot_idx)
            key = (next_budget, next_len)
            incumbent = updates.get(key)
            if incumbent is None or next_overlap > incumbent[0]:
                updates[key] = (next_overlap, next_mask)
            elif next_overlap == incumbent[0]:
                if popcount(next_mask) < popcount(incumbent[1]):
                    updates[key] = (next_overlap, next_mask)
        states = updates

    best = (0, 0, 0)
    best_budget = 0
    for (budget, selected_len), (overlap, mask) in states.items():
        candidate = (int(overlap), int(selected_len), int(mask))
        if better_state(candidate, best, gt_len):
            best = candidate
            best_budget = int(budget)

    overlap, selected_frames, mask = best
    f1 = 0.0 if selected_frames <= 0 else 2.0 * overlap / (selected_frames + gt_len)
    return {
        'f1': float(f1),
        'mask': int(mask),
        'overlap': int(overlap),
        'budget_weight': int(best_budget),
        'selected_frames': int(selected_frames),
        'gt_frames': int(gt_len),
    }


def exact_oracle_max_user(
    user_summary: np.ndarray,
    cps: np.ndarray,
    nfps: np.ndarray,
    n_frames: int,
    capacity: int,
) -> Dict:
    users = np.asarray(user_summary, dtype=bool)
    if users.ndim != 2:
        raise ValueError(f'Expected user_summary shape [U, n_frames], got {users.shape}')
    best = None
    for user_idx in range(users.shape[0]):
        result = exact_oracle_for_user(
            user_summary=users[user_idx],
            cps=cps,
            nfps=nfps,
            n_frames=n_frames,
            capacity=capacity,
        )
        result['best_user_idx'] = int(user_idx)
        if best is None:
            best = result
            continue
        candidate = (result['overlap'], result['selected_frames'], result['mask'])
        incumbent = (best['overlap'], best['selected_frames'], best['mask'])
        if result['f1'] > best['f1'] + 1e-12 or (
            abs(result['f1'] - best['f1']) <= 1e-12
            and better_state(candidate, incumbent, int(result['gt_frames']))
        ):
            best = result
    if best is None:
        raise ValueError('No users available for oracle search.')
    return best


def binary_f1(a: np.ndarray, b: np.ndarray) -> float:
    return vsumm_helper.f1_score(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))


def open_group(key: str, cache: Dict[str, h5py.File]):
    key_path = Path(key)
    h5_path = str(key_path.parent)
    h5_key = key_path.name
    if h5_path not in cache:
        cache[h5_path] = h5py.File(h5_path, 'r')
    return h5_key, cache[h5_path][h5_key]


def analyze_video(key: str, group, dataset: str, summary_budget: float) -> Dict:
    h5_key = str(Path(key).name)
    gtscore = normalize_gtscore(group['gtscore'][...].astype(np.float32))
    user_summary = group['user_summary'][...].astype(np.float32)
    cps = group['change_points'][...].astype(np.int32)
    n_frames = scalar_int(group['n_frames'][...])
    nfps = group['n_frame_per_seg'][...].astype(np.int32).reshape(-1)
    picks = group['picks'][...].astype(np.int32).reshape(-1)

    if gtscore.shape[0] != picks.shape[0]:
        raise ValueError(f'gtscore/picks mismatch for {key}: {gtscore.shape[0]} vs {picks.shape[0]}')
    if cps.shape[0] != nfps.shape[0]:
        raise ValueError(f'cps/nfps mismatch for {key}: {cps.shape[0]} vs {nfps.shape[0]}')

    capacity = int(n_frames * float(summary_budget))
    eval_metric = infer_f1_metric_from_dataset(dataset)

    gtscore_summary = vsumm_helper.get_keyshot_summ(
        gtscore,
        cps,
        n_frames,
        nfps,
        picks,
        proportion=summary_budget,
    )
    gtscore_summary_f1 = vsumm_helper.get_summ_f1score(
        pred_summ=gtscore_summary,
        test_summ=user_summary,
        eval_metric=eval_metric,
    )
    gtscore_summary_scores = summary_to_sample_scores(gtscore_summary, picks, n_frames)
    gtscore_summary_tau, gtscore_summary_rho = rank_or_nan(
        gtscore_summary_scores,
        gtscore,
        key=f'{key}:gtscore_knapsack_summary',
    )

    oracle = exact_oracle_max_user(
        user_summary=user_summary,
        cps=cps,
        nfps=nfps,
        n_frames=n_frames,
        capacity=capacity,
    )
    oracle_summary = build_summary_from_bitset(oracle['mask'], cps, n_frames)
    oracle_f1_eval = vsumm_helper.get_summ_f1score(
        pred_summ=oracle_summary,
        test_summ=user_summary,
        eval_metric=eval_metric,
    )
    oracle_scores = summary_to_sample_scores(oracle_summary, picks, n_frames)
    oracle_tau, oracle_rho = rank_or_nan(oracle_scores, gtscore, key=f'{key}:oracle_max_f1_summary')

    shot_lengths = (cps[:, 1] - cps[:, 0] + 1).astype(np.int32)
    gtscore_selected_shots = int(sum(int(gtscore_summary[int(first):int(last) + 1].any()) for first, last in cps))
    oracle_selected_shots = int(popcount(oracle['mask']))

    f1_gap = float(oracle_f1_eval - gtscore_summary_f1)
    return {
        'key': key,
        'h5_key': h5_key,
        'n_frames': int(n_frames),
        'num_sampled_frames': int(picks.shape[0]),
        'num_shots': int(cps.shape[0]),
        'num_users': int(np.asarray(user_summary).shape[0]),
        'budget_frames': int(capacity),
        'summary_budget': float(summary_budget),
        'nfps_matches_cps_lengths': bool(np.array_equal(nfps, shot_lengths)),
        'gtscore_knapsack_f1': float(gtscore_summary_f1),
        'gtscore_knapsack_tau': float(gtscore_summary_tau),
        'gtscore_knapsack_rho': float(gtscore_summary_rho),
        'gtscore_knapsack_budget_ratio': float(np.asarray(gtscore_summary, dtype=bool).sum() / max(1, n_frames)),
        'gtscore_selected_shots': gtscore_selected_shots,
        'oracle_max_f1': float(oracle_f1_eval),
        'oracle_dp_f1_best_user': float(oracle['f1']),
        'oracle_tau_vs_gtscore': float(oracle_tau),
        'oracle_rho_vs_gtscore': float(oracle_rho),
        'oracle_budget_ratio': float(np.asarray(oracle_summary, dtype=bool).sum() / max(1, n_frames)),
        'oracle_budget_weight_ratio': float(int(oracle['budget_weight']) / max(1, n_frames)),
        'oracle_selected_frames': int(np.asarray(oracle_summary, dtype=bool).sum()),
        'oracle_selected_shots': oracle_selected_shots,
        'oracle_best_user_idx': int(oracle['best_user_idx']),
        'oracle_overlap_frames': int(oracle['overlap']),
        'oracle_gt_user_frames': int(oracle['gt_frames']),
        'oracle_vs_gtscore_summary_f1': float(binary_f1(oracle_summary, gtscore_summary)),
        'oracle_minus_gtscore_f1': f1_gap,
        'low_oracle_rank_tau_lt_0_10': bool(np.isfinite(oracle_tau) and oracle_tau < 0.10),
        'low_oracle_rank_tau_le_0': bool(np.isfinite(oracle_tau) and oracle_tau <= 0.0),
        'target_conflict_gap_gt_0_05_tau_lt_0_10': bool(
            f1_gap > 0.05 and np.isfinite(oracle_tau) and oracle_tau < 0.10
        ),
    }


def quantile(values: Iterable[float], q: float) -> float:
    finite = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float('nan')
    return float(np.quantile(finite, q))


def summarize(records: List[Dict]) -> Dict:
    def vals(name: str) -> List[float]:
        return [float(record[name]) for record in records]

    if not records:
        return {'num_videos': 0}

    oracle_tau = vals('oracle_tau_vs_gtscore')
    oracle_rho = vals('oracle_rho_vs_gtscore')
    f1_gap = vals('oracle_minus_gtscore_f1')
    return {
        'num_videos': int(len(records)),
        'mean_gtscore_knapsack_f1': safe_nanmean(vals('gtscore_knapsack_f1'), default=float('nan')),
        'mean_gtscore_knapsack_tau': safe_nanmean(vals('gtscore_knapsack_tau'), default=float('nan')),
        'mean_gtscore_knapsack_rho': safe_nanmean(vals('gtscore_knapsack_rho'), default=float('nan')),
        'mean_oracle_max_f1': safe_nanmean(vals('oracle_max_f1'), default=float('nan')),
        'mean_oracle_tau_vs_gtscore': safe_nanmean(oracle_tau, default=float('nan')),
        'mean_oracle_rho_vs_gtscore': safe_nanmean(oracle_rho, default=float('nan')),
        'median_oracle_tau_vs_gtscore': quantile(oracle_tau, 0.50),
        'q25_oracle_tau_vs_gtscore': quantile(oracle_tau, 0.25),
        'q75_oracle_tau_vs_gtscore': quantile(oracle_tau, 0.75),
        'mean_oracle_vs_gtscore_summary_f1': safe_nanmean(vals('oracle_vs_gtscore_summary_f1'), default=float('nan')),
        'mean_oracle_minus_gtscore_f1': safe_nanmean(f1_gap, default=float('nan')),
        'fraction_oracle_tau_lt_0_10': float(np.mean([record['low_oracle_rank_tau_lt_0_10'] for record in records])),
        'fraction_oracle_tau_le_0': float(np.mean([record['low_oracle_rank_tau_le_0'] for record in records])),
        'fraction_conflict_gap_gt_0_05_tau_lt_0_10': float(
            np.mean([record['target_conflict_gap_gt_0_05_tau_lt_0_10'] for record in records])
        ),
        'mean_oracle_budget_ratio': safe_nanmean(vals('oracle_budget_ratio'), default=float('nan')),
        'mean_gtscore_knapsack_budget_ratio': safe_nanmean(
            vals('gtscore_knapsack_budget_ratio'), default=float('nan')
        ),
    }


def write_csv(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(records[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    args = parse_args()
    splits = load_all_splits(args.splits, val_ratio=args.val_ratio, seed=args.seed)
    validate_splits(splits, expected_dataset=args.dataset)
    keys = collect_unique_keys(splits, args.selection_part)
    if args.max_videos is not None:
        keys = keys[:args.max_videos]
    if not keys:
        raise ValueError('No keys selected for diagnostic.')

    cache: Dict[str, h5py.File] = {}
    records: List[Dict] = []
    try:
        for key in keys:
            _h5_key, group = open_group(key, cache)
            records.append(
                analyze_video(
                    key=key,
                    group=group,
                    dataset=args.dataset,
                    summary_budget=args.summary_budget,
                )
            )
    finally:
        for handle in cache.values():
            handle.close()

    records = sorted(records, key=lambda record: record['h5_key'])
    summary = summarize(records)
    output = {
        'meta': {
            'dataset': args.dataset,
            'splits': args.splits,
            'selection_part': args.selection_part,
            'val_ratio': float(args.val_ratio),
            'seed': int(args.seed),
            'summary_budget': float(args.summary_budget),
            'diagnostic_only': True,
            'uses_human_labels_for_training': False,
            'human_label_use': 'offline protocol diagnostic only',
            'oracle_definition': 'exact per-user budgeted keyshot subset maximizing SumMe max-user F1, then max over users',
            'rank_definition': 'binary oracle/gtscore-knapsack sampled scores compared with aggregated HDF5 gtscore',
        },
        'summary': summary,
        'records': records,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    write_csv(Path(args.output_csv), records)

    print(json.dumps({'summary': summary, 'output_json': str(output_json), 'output_csv': args.output_csv}, indent=2))


if __name__ == '__main__':
    main()