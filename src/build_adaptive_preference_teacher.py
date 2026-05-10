# -*- coding: utf-8 -*-
"""Build a reliability-gated preference teacher from existing weak teachers.

The builder combines a source preference teacher with a non-human shot utility
teacher. The source teacher's pair count is used only as a confidence proxy:
videos with enough source preference pairs receive a stronger language-prior
blend; low-pair videos fall back to a conservative blend.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from helpers.preference_teacher_helper import (
    PreferenceTeacherStore,
    build_preference_teacher_record,
    normalize_01,
)
from helpers.shot_utility_helper import ShotUtilityStore


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='summe', choices=('summe', 'tvsum'))
    parser.add_argument('--h5-path', type=str, required=True)
    parser.add_argument('--source-teacher-path', type=str, required=True)
    parser.add_argument('--shot-utility-path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--output-meta-json', type=str, default=None)
    parser.add_argument('--baseline-formula', type=str, default='phase1_default')
    parser.add_argument('--pair-count-threshold', type=int, default=32)
    parser.add_argument('--low-llm-weight', type=float, default=0.10)
    parser.add_argument('--high-llm-weight', type=float, default=0.70)
    parser.add_argument('--num-perspectives', type=int, default=7)
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--max-pairs-per-video', type=int, default=96)
    parser.add_argument('--pair-seed', type=int, default=19500)
    parser.add_argument('--positive-threshold', type=float, default=0.60)
    parser.add_argument('--negative-threshold', type=float, default=0.20)
    parser.add_argument('--limit', type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.pair_count_threshold < 0:
        raise ValueError('--pair-count-threshold must be >= 0')
    for name in ('low_llm_weight', 'high_llm_weight'):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f'--{name.replace("_", "-")} must be in [0, 1]')
    if args.low_llm_weight > args.high_llm_weight:
        raise ValueError('--low-llm-weight must be <= --high-llm-weight')
    if args.num_perspectives <= 0:
        raise ValueError('--num-perspectives must be positive')


def build_adaptive_scores(
    base_scores: np.ndarray,
    source_scores: np.ndarray,
    source_pair_count: int,
    pair_count_threshold: int,
    low_llm_weight: float,
    high_llm_weight: float,
) -> Dict:
    base = normalize_01(base_scores).reshape(-1)
    source = normalize_01(source_scores).reshape(-1)
    if base.shape[0] != source.shape[0]:
        raise ValueError(
            f'score length mismatch: base={base.shape[0]} source={source.shape[0]}'
        )
    use_high = int(source_pair_count) >= int(pair_count_threshold)
    weight = float(high_llm_weight if use_high else low_llm_weight)
    scores = normalize_01((1.0 - weight) * base + weight * source)
    return {
        'scores': scores.astype(np.float32),
        'llm_weight': weight,
        'use_high_weight': bool(use_high),
    }


def build_records(args: argparse.Namespace) -> Dict[str, Dict]:
    source_store = PreferenceTeacherStore(Path(args.source_teacher_path))
    utility_store = ShotUtilityStore(Path(args.shot_utility_path))
    records: Dict[str, Dict] = {}
    stats: List[Dict] = []

    with h5py.File(args.h5_path, 'r') as h5:
        keys = sorted(h5.keys())
        if args.limit is not None:
            keys = keys[:int(args.limit)]
        for h5_key in keys:
            group = h5[h5_key]
            cps = group['change_points'][...].astype(np.int32)
            nfps = group['n_frame_per_seg'][...].astype(np.int32)
            n_frames = int(np.asarray(group['n_frames'][...]).item())
            source_record = source_store.get(h5_key)
            source_scores = np.asarray(source_record['shot_scores'], dtype=np.float32).reshape(-1)
            base_scores = utility_store.get(
                h5_key=h5_key,
                formula_name=args.baseline_formula,
            )
            if source_scores.shape[0] != cps.shape[0]:
                raise ValueError(
                    f'source score length mismatch for {h5_key}: '
                    f'{source_scores.shape[0]} vs {cps.shape[0]}'
                )
            source_pair_count = int(np.asarray(source_record['pair_i']).reshape(-1).shape[0])
            adaptive = build_adaptive_scores(
                base_scores=base_scores,
                source_scores=source_scores,
                source_pair_count=source_pair_count,
                pair_count_threshold=args.pair_count_threshold,
                low_llm_weight=args.low_llm_weight,
                high_llm_weight=args.high_llm_weight,
            )
            perspective_scores = np.tile(
                adaptive['scores'].reshape(1, -1),
                (int(args.num_perspectives), 1),
            ).astype(np.float32)
            meta = {
                'dataset': args.dataset,
                'h5_key': h5_key,
                'source_teacher_path': args.source_teacher_path,
                'shot_utility_path': args.shot_utility_path,
                'baseline_formula': args.baseline_formula,
                'calibration': 'pair_count_adaptive_blend',
                'source_pair_count': source_pair_count,
                'pair_count_threshold': int(args.pair_count_threshold),
                'llm_weight': float(adaptive['llm_weight']),
                'use_high_weight': bool(adaptive['use_high_weight']),
                'low_llm_weight': float(args.low_llm_weight),
                'high_llm_weight': float(args.high_llm_weight),
                'num_perspectives': int(args.num_perspectives),
                'summary_budget': float(args.summary_budget),
            }
            record = build_preference_teacher_record(
                perspective_scores=perspective_scores,
                nfps=nfps,
                n_frames=n_frames,
                summary_budget=float(args.summary_budget),
                max_pairs_per_video=int(args.max_pairs_per_video),
                pair_seed=int(args.pair_seed),
                positive_threshold=float(args.positive_threshold),
                negative_threshold=float(args.negative_threshold),
                meta=meta,
            )
            records[h5_key] = record
            stats.append({
                'h5_key': h5_key,
                'num_shots': int(cps.shape[0]),
                'source_pair_count': source_pair_count,
                'llm_weight': float(adaptive['llm_weight']),
                'use_high_weight': bool(adaptive['use_high_weight']),
                'num_pairs': int(np.asarray(record['pair_i']).reshape(-1).shape[0]),
                'num_positive_shots': int((record['inclusion_prob'] >= args.positive_threshold).sum()),
                'num_negative_shots': int((record['inclusion_prob'] <= args.negative_threshold).sum()),
            })

    build_records.last_stats = stats  # type: ignore[attr-defined]
    return records


def summarize_stats(stats: List[Dict]) -> Dict:
    if not stats:
        return {'num_videos': 0}
    return {
        'num_videos': int(len(stats)),
        'num_high_weight_videos': int(sum(1 for row in stats if row['use_high_weight'])),
        'num_low_weight_videos': int(sum(1 for row in stats if not row['use_high_weight'])),
        'mean_source_pair_count': float(np.mean([row['source_pair_count'] for row in stats])),
        'mean_output_pair_count': float(np.mean([row['num_pairs'] for row in stats])),
        'min_output_pair_count': int(min(row['num_pairs'] for row in stats)),
        'mean_positive_shots': float(np.mean([row['num_positive_shots'] for row in stats])),
        'mean_negative_shots': float(np.mean([row['num_negative_shots'] for row in stats])),
    }


def main() -> None:
    args = get_parser().parse_args()
    validate_args(args)
    records = build_records(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, records)

    stats = getattr(build_records, 'last_stats', [])
    meta = {
        'dataset': args.dataset,
        'output': str(output_path),
        'source_teacher_path': args.source_teacher_path,
        'shot_utility_path': args.shot_utility_path,
        'baseline_formula': args.baseline_formula,
        'calibration': 'pair_count_adaptive_blend',
        'pair_count_threshold': int(args.pair_count_threshold),
        'low_llm_weight': float(args.low_llm_weight),
        'high_llm_weight': float(args.high_llm_weight),
        'summary': summarize_stats(stats),
        'records': stats,
    }
    if args.output_meta_json:
        meta_path = Path(args.output_meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in meta.items() if k != 'records'}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
