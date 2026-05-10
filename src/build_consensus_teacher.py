# -*- coding: utf-8 -*-
"""Build a consensus preference teacher from multiple weak teachers.

This builder does not average teacher scores. Each source teacher first casts a
shot-level vote: positive, negative, or uncertain. The output teacher supervises
only shots where teachers agree on positive or negative; all remaining shots are
marked uncertain with zero confidence so inclusion/pair losses ignore them.

The script does not read human evaluation fields or human summaries.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from helpers.preference_teacher_helper import (
    PreferenceTeacherStore,
    build_preference_pairs,
    validate_preference_record,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='summe', choices=('summe', 'tvsum'))
    parser.add_argument('--h5-path', type=str, required=True)
    parser.add_argument('--teacher-paths', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--output-meta-json', type=str, default=None)
    parser.add_argument('--positive-threshold', type=float, default=0.60)
    parser.add_argument('--negative-threshold', type=float, default=0.20)
    parser.add_argument(
        '--min-agree-count',
        type=int,
        default=None,
        help='Minimum same-side votes required. Default requires all teachers.',
    )
    parser.add_argument(
        '--allow-neutral-votes',
        action='store_true',
        help='Allow uncertain source votes when min-agree-count is met and there are no opposite votes.',
    )
    parser.add_argument('--max-pairs-per-video', type=int, default=96)
    parser.add_argument('--pair-seed', type=int, default=19500)
    parser.add_argument('--limit', type=int, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if len(args.teacher_paths) < 2:
        raise ValueError('--teacher-paths requires at least two teacher files.')
    if not (0.0 <= float(args.negative_threshold) < float(args.positive_threshold) <= 1.0):
        raise ValueError('Expected 0 <= negative_threshold < positive_threshold <= 1.')
    if args.max_pairs_per_video <= 0:
        raise ValueError('--max-pairs-per-video must be positive.')
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive when provided.')
    if args.min_agree_count is not None:
        if args.min_agree_count <= 0:
            raise ValueError('--min-agree-count must be positive when provided.')
        if args.min_agree_count > len(args.teacher_paths):
            raise ValueError('--min-agree-count cannot exceed number of teacher files.')


def load_teacher_records(stores: Sequence[PreferenceTeacherStore], h5_key: str) -> List[Dict]:
    records = []
    for store in stores:
        records.append(store.get(h5_key))
    return records


def validate_lengths(h5_key: str, num_shots: int, records: Sequence[Dict]) -> None:
    for idx, record in enumerate(records):
        length = int(np.asarray(record['inclusion_prob']).reshape(-1).shape[0])
        if length != int(num_shots):
            raise ValueError(
                f'Teacher {idx} inclusion length mismatch for {h5_key}: '
                f'{length} vs expected {num_shots}'
            )


def classify_votes(
    records: Sequence[Dict],
    positive_threshold: float,
    negative_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    votes = []
    for record in records:
        inclusion = np.asarray(record['inclusion_prob'], dtype=np.float32).reshape(-1)
        vote = np.zeros_like(inclusion, dtype=np.int8)
        vote[inclusion >= float(positive_threshold)] = 1
        vote[inclusion <= float(negative_threshold)] = -1
        votes.append(vote)
    vote_matrix = np.stack(votes, axis=0)
    pos_votes = (vote_matrix == 1).sum(axis=0).astype(np.int32)
    neg_votes = (vote_matrix == -1).sum(axis=0).astype(np.int32)
    uncertain_votes = (vote_matrix == 0).sum(axis=0).astype(np.int32)
    return pos_votes, neg_votes, uncertain_votes


def consensus_labels(
    pos_votes: np.ndarray,
    neg_votes: np.ndarray,
    num_teachers: int,
    min_agree_count: Optional[int],
    allow_neutral_votes: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = int(num_teachers if min_agree_count is None else min_agree_count)
    if allow_neutral_votes:
        agree_positive = (pos_votes >= required) & (neg_votes == 0)
        agree_negative = (neg_votes >= required) & (pos_votes == 0)
    else:
        agree_positive = pos_votes == int(num_teachers)
        agree_negative = neg_votes == int(num_teachers)
    uncertain = ~(agree_positive | agree_negative)
    return agree_positive, agree_negative, uncertain


def build_consensus_record(
    h5_key: str,
    num_shots: int,
    records: Sequence[Dict],
    teacher_paths: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Dict, Dict]:
    pos_votes, neg_votes, uncertain_votes = classify_votes(
        records=records,
        positive_threshold=float(args.positive_threshold),
        negative_threshold=float(args.negative_threshold),
    )
    agree_positive, agree_negative, uncertain = consensus_labels(
        pos_votes=pos_votes,
        neg_votes=neg_votes,
        num_teachers=len(records),
        min_agree_count=args.min_agree_count,
        allow_neutral_votes=bool(args.allow_neutral_votes),
    )

    inclusion_prob = np.full((num_shots,), 0.5, dtype=np.float32)
    inclusion_prob[agree_positive] = 1.0
    inclusion_prob[agree_negative] = 0.0

    teacher_confidence = np.zeros((num_shots,), dtype=np.float32)
    teacher_confidence[agree_positive | agree_negative] = 1.0

    shot_scores = inclusion_prob.copy()
    summary_masks = agree_positive.reshape(1, -1).astype(np.float32)

    pair_i, pair_j, pair_label, pair_confidence = build_preference_pairs(
        inclusion_prob=inclusion_prob,
        positive_threshold=float(args.positive_threshold),
        negative_threshold=float(args.negative_threshold),
        max_pairs_per_video=int(args.max_pairs_per_video),
        pair_seed=int(args.pair_seed),
    )

    meta = {
        'dataset': args.dataset,
        'h5_key': h5_key,
        'teacher_paths': list(teacher_paths),
        'calibration': 'consensus_agree_pos_neg_ignore_uncertain',
        'positive_threshold': float(args.positive_threshold),
        'negative_threshold': float(args.negative_threshold),
        'min_agree_count': int(len(records) if args.min_agree_count is None else args.min_agree_count),
        'allow_neutral_votes': bool(args.allow_neutral_votes),
        'num_teachers': int(len(records)),
        'max_pairs_per_video': int(args.max_pairs_per_video),
        'pair_seed': int(args.pair_seed),
        'recommended_training_note': 'Set lambda_pref_list=0 so uncertain shots are fully ignored by training.',
    }
    record = {
        'shot_scores': shot_scores.astype(np.float32),
        'inclusion_prob': inclusion_prob.astype(np.float32),
        'pair_i': pair_i.astype(np.int64),
        'pair_j': pair_j.astype(np.int64),
        'pair_label': pair_label.astype(np.float32),
        'pair_confidence': pair_confidence.astype(np.float32),
        'summary_masks': summary_masks.astype(np.float32),
        'teacher_confidence': teacher_confidence.astype(np.float32),
        'meta': meta,
    }
    record = validate_preference_record(record, h5_key=h5_key)
    row = {
        'h5_key': h5_key,
        'num_shots': int(num_shots),
        'num_agree_positive': int(agree_positive.sum()),
        'num_agree_negative': int(agree_negative.sum()),
        'num_uncertain': int(uncertain.sum()),
        'agreement_rate': float(teacher_confidence.mean()) if num_shots > 0 else 0.0,
        'num_pairs': int(pair_i.shape[0]),
        'mean_positive_votes': float(pos_votes.mean()) if num_shots > 0 else 0.0,
        'mean_negative_votes': float(neg_votes.mean()) if num_shots > 0 else 0.0,
        'mean_uncertain_votes': float(uncertain_votes.mean()) if num_shots > 0 else 0.0,
    }
    return record, row


def build_records(args: argparse.Namespace) -> Dict[str, Dict]:
    stores = [PreferenceTeacherStore(Path(path)) for path in args.teacher_paths]
    records: Dict[str, Dict] = {}
    stats: List[Dict] = []

    with h5py.File(args.h5_path, 'r') as h5:
        keys = sorted(h5.keys())
        if args.limit is not None:
            keys = keys[:int(args.limit)]
        for h5_key in keys:
            num_shots = int(h5[h5_key]['change_points'].shape[0])
            source_records = load_teacher_records(stores, h5_key)
            validate_lengths(h5_key, num_shots, source_records)
            record, row = build_consensus_record(
                h5_key=h5_key,
                num_shots=num_shots,
                records=source_records,
                teacher_paths=args.teacher_paths,
                args=args,
            )
            records[h5_key] = record
            stats.append(row)

    build_records.last_stats = stats  # type: ignore[attr-defined]
    return records


def summarize_stats(stats: Sequence[Dict]) -> Dict:
    if not stats:
        return {'num_videos': 0}
    return {
        'num_videos': int(len(stats)),
        'mean_agree_positive': float(np.mean([row['num_agree_positive'] for row in stats])),
        'mean_agree_negative': float(np.mean([row['num_agree_negative'] for row in stats])),
        'mean_uncertain': float(np.mean([row['num_uncertain'] for row in stats])),
        'mean_agreement_rate': float(np.mean([row['agreement_rate'] for row in stats])),
        'mean_num_pairs': float(np.mean([row['num_pairs'] for row in stats])),
        'min_num_pairs': int(min(row['num_pairs'] for row in stats)),
        'max_num_pairs': int(max(row['num_pairs'] for row in stats)),
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
        'teacher_paths': list(args.teacher_paths),
        'calibration': 'consensus_agree_pos_neg_ignore_uncertain',
        'positive_threshold': float(args.positive_threshold),
        'negative_threshold': float(args.negative_threshold),
        'min_agree_count': int(len(args.teacher_paths) if args.min_agree_count is None else args.min_agree_count),
        'allow_neutral_votes': bool(args.allow_neutral_votes),
        'recommended_training_note': 'Use LAMBDA_PREF_LIST=0 for this teacher with current preference_distill code.',
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