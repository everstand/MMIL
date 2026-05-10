# -*- coding: utf-8 -*-
"""Preference teacher utilities for shot-level weak supervision.

Records produced by this helper are built from caption/metadata weak evidence
and contain shot-level preference scores, budgeted masks, and deterministic
pairwise preferences for optional training.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from helpers import vsumm_helper


CANONICAL_PERSPECTIVE_IDS = (
    'storyline_temporal_progression',
    'visual_diversity',
    'human_interest_memorability',
    'action_event_transition',
    'visual_scene_object_coverage',
    'anti_redundancy_background_suppression',
    'concise_coherent_skimming',
)


def normalize_01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('normalize_01 received non-finite values.')

    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo + eps)).astype(np.float32)


def _as_1d_float(name: str,
                 value,
                 length: Optional[int] = None,
                 lower: Optional[float] = None,
                 upper: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if length is not None and arr.shape[0] != int(length):
        raise ValueError(f'{name} length mismatch: {arr.shape[0]} vs expected {length}')
    if not np.isfinite(arr).all():
        raise ValueError(f'{name} contains non-finite values.')
    if lower is not None and float(arr.min(initial=lower)) < lower - 1e-6:
        raise ValueError(f'{name} contains values below {lower}.')
    if upper is not None and float(arr.max(initial=upper)) > upper + 1e-6:
        raise ValueError(f'{name} contains values above {upper}.')
    return arr.astype(np.float32)


def _as_1d_int(name: str, value, length: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if length is not None and arr.shape[0] != int(length):
        raise ValueError(f'{name} length mismatch: {arr.shape[0]} vs expected {length}')
    return arr.astype(np.int64)


def validate_perspective_scores(perspective_scores: np.ndarray,
                                expected_num_perspectives: Optional[int] = None,
                                expected_num_shots: Optional[int] = None) -> np.ndarray:
    scores = np.asarray(perspective_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f'Expected perspective_scores shape [K, S], got {scores.shape}')
    if expected_num_perspectives is not None and scores.shape[0] != int(expected_num_perspectives):
        raise ValueError(
            f'perspective count mismatch: {scores.shape[0]} vs {expected_num_perspectives}'
        )
    if expected_num_shots is not None and scores.shape[1] != int(expected_num_shots):
        raise ValueError(f'shot count mismatch: {scores.shape[1]} vs {expected_num_shots}')
    if scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError(f'Invalid empty perspective_scores shape: {scores.shape}')
    if not np.isfinite(scores).all():
        raise ValueError('perspective_scores contains non-finite values.')
    if scores.min() < -1e-6 or scores.max() > 1.0 + 1e-6:
        raise ValueError('perspective_scores must be in [0, 1].')
    return np.clip(scores, 0.0, 1.0).astype(np.float32)


def build_summary_masks_from_perspective_scores(perspective_scores: np.ndarray,
                                                 nfps: np.ndarray,
                                                 n_frames: int,
                                                 summary_budget: float) -> np.ndarray:
    scores = validate_perspective_scores(perspective_scores)
    nfps = np.asarray(nfps, dtype=np.int32).reshape(-1)
    if nfps.shape[0] != scores.shape[1]:
        raise ValueError(f'nfps length mismatch: {nfps.shape[0]} vs scores S={scores.shape[1]}')
    if not np.isfinite(nfps).all() or (nfps <= 0).any():
        raise ValueError('nfps must contain positive finite shot lengths.')
    if int(n_frames) <= 0:
        raise ValueError(f'Invalid n_frames={n_frames}')
    if not (0.0 < float(summary_budget) < 1.0):
        raise ValueError(f'Invalid summary_budget={summary_budget}; expected 0 < budget < 1.')

    capacity = int(int(n_frames) * float(summary_budget))
    masks = np.zeros(scores.shape, dtype=np.float32)
    if capacity <= 0:
        return masks

    for k in range(scores.shape[0]):
        scores_k = normalize_01(scores[k])
        values = np.round(scores_k * 1000.0).astype(np.int32)
        if values.size == 0 or int(values.max(initial=0)) <= 0:
            selected_idx = []
        else:
            selected_idx = vsumm_helper.knapsack(values.tolist(), nfps.tolist(), capacity)
        if selected_idx:
            masks[k, np.asarray(selected_idx, dtype=np.int64)] = 1.0
    return masks.astype(np.float32)


def build_preference_pairs(inclusion_prob: np.ndarray,
                           positive_threshold: float = 0.60,
                           negative_threshold: float = 0.20,
                           max_pairs_per_video: int = 96,
                           pair_seed: int = 19500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inclusion = _as_1d_float('inclusion_prob', inclusion_prob, lower=0.0, upper=1.0)
    if not (0.0 <= positive_threshold <= 1.0):
        raise ValueError(f'Invalid positive_threshold={positive_threshold}')
    if not (0.0 <= negative_threshold <= 1.0):
        raise ValueError(f'Invalid negative_threshold={negative_threshold}')
    if positive_threshold <= negative_threshold:
        raise ValueError(
            f'Expected positive_threshold > negative_threshold, got '
            f'{positive_threshold} <= {negative_threshold}'
        )
    if max_pairs_per_video <= 0:
        raise ValueError(f'Invalid max_pairs_per_video={max_pairs_per_video}; expected > 0.')

    positive = np.where(inclusion >= float(positive_threshold))[0]
    negative = np.where(inclusion <= float(negative_threshold))[0]
    if positive.size == 0 or negative.size == 0:
        empty_i = np.empty((0,), dtype=np.int64)
        empty_f = np.empty((0,), dtype=np.float32)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy()

    rng = np.random.RandomState(int(pair_seed))
    rows = []
    for i in positive.tolist():
        for j in negative.tolist():
            gap = abs(float(inclusion[i]) - float(inclusion[j]))
            rows.append((gap, float(rng.uniform()), int(i), int(j)))

    rows.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    rows = rows[:int(max_pairs_per_video)]

    pair_i = np.asarray([row[2] for row in rows], dtype=np.int64)
    pair_j = np.asarray([row[3] for row in rows], dtype=np.int64)
    pair_label = np.ones((len(rows),), dtype=np.float32)
    pair_confidence = np.asarray([row[0] for row in rows], dtype=np.float32)
    return pair_i, pair_j, pair_label, np.clip(pair_confidence, 0.0, 1.0).astype(np.float32)


def build_preference_teacher_record(perspective_scores: np.ndarray,
                                    nfps: np.ndarray,
                                    n_frames: int,
                                    summary_budget: float,
                                    max_pairs_per_video: int = 96,
                                    pair_seed: int = 19500,
                                    positive_threshold: float = 0.60,
                                    negative_threshold: float = 0.20,
                                    meta: Optional[Dict] = None) -> Dict:
    scores = validate_perspective_scores(perspective_scores)
    summary_masks = build_summary_masks_from_perspective_scores(
        perspective_scores=scores,
        nfps=nfps,
        n_frames=int(n_frames),
        summary_budget=float(summary_budget),
    )
    inclusion_prob = summary_masks.mean(axis=0).astype(np.float32)
    shot_scores = normalize_01(scores.mean(axis=0)).astype(np.float32)
    teacher_confidence = np.maximum(inclusion_prob, 1.0 - inclusion_prob).astype(np.float32)
    pair_i, pair_j, pair_label, pair_confidence = build_preference_pairs(
        inclusion_prob=inclusion_prob,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        max_pairs_per_video=max_pairs_per_video,
        pair_seed=pair_seed,
    )

    record = {
        'shot_scores': shot_scores,
        'inclusion_prob': inclusion_prob,
        'pair_i': pair_i,
        'pair_j': pair_j,
        'pair_label': pair_label,
        'pair_confidence': pair_confidence,
        'summary_masks': summary_masks,
        'teacher_confidence': teacher_confidence,
        'meta': dict(meta or {}),
    }
    return validate_preference_record(record, h5_key=str(record['meta'].get('h5_key', '<unknown>')))


def validate_preference_record(record: Dict, h5_key: str = '<unknown>') -> Dict:
    if not isinstance(record, dict):
        raise ValueError(f'Preference teacher record for {h5_key} must be dict, got {type(record)}')

    required = (
        'shot_scores',
        'inclusion_prob',
        'pair_i',
        'pair_j',
        'pair_label',
        'pair_confidence',
        'summary_masks',
        'teacher_confidence',
        'meta',
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise KeyError(f'Missing preference teacher fields for {h5_key}: {missing}')

    shot_scores = _as_1d_float('shot_scores', record['shot_scores'])
    if shot_scores.size <= 0:
        raise ValueError(f'Empty shot_scores for {h5_key}')
    num_shots = int(shot_scores.shape[0])

    inclusion_prob = _as_1d_float('inclusion_prob', record['inclusion_prob'], num_shots, 0.0, 1.0)
    teacher_confidence = _as_1d_float(
        'teacher_confidence', record['teacher_confidence'], num_shots, 0.0, 1.0
    )

    summary_masks = np.asarray(record['summary_masks'], dtype=np.float32)
    if summary_masks.ndim != 2:
        raise ValueError(f'summary_masks for {h5_key} must be [K, S], got {summary_masks.shape}')
    if summary_masks.shape[1] != num_shots:
        raise ValueError(
            f'summary_masks shot count mismatch for {h5_key}: '
            f'{summary_masks.shape[1]} vs {num_shots}'
        )
    if summary_masks.shape[0] <= 0:
        raise ValueError(f'summary_masks has no perspectives for {h5_key}')
    if not np.isfinite(summary_masks).all():
        raise ValueError(f'summary_masks contains non-finite values for {h5_key}')
    if ((summary_masks < -1e-6) | (summary_masks > 1.0 + 1e-6)).any():
        raise ValueError(f'summary_masks must be binary/soft values in [0, 1] for {h5_key}')
    if not np.all(np.isclose(summary_masks, np.round(summary_masks), atol=1e-6)):
        raise ValueError(f'summary_masks must be binary for {h5_key}')
    summary_masks = np.round(summary_masks).astype(np.float32)

    pair_i = _as_1d_int('pair_i', record['pair_i'])
    pair_j = _as_1d_int('pair_j', record['pair_j'], length=pair_i.shape[0])
    pair_label = _as_1d_float('pair_label', record['pair_label'], pair_i.shape[0], 0.0, 1.0)
    pair_confidence = _as_1d_float(
        'pair_confidence', record['pair_confidence'], pair_i.shape[0], 0.0, 1.0
    )
    if pair_i.size > 0:
        if int(pair_i.min()) < 0 or int(pair_i.max()) >= num_shots:
            raise ValueError(f'pair_i index out of range for {h5_key}')
        if int(pair_j.min()) < 0 or int(pair_j.max()) >= num_shots:
            raise ValueError(f'pair_j index out of range for {h5_key}')

    meta = record['meta']
    if not isinstance(meta, dict):
        raise ValueError(f'meta for {h5_key} must be dict, got {type(meta)}')

    return {
        'shot_scores': shot_scores.astype(np.float32),
        'inclusion_prob': inclusion_prob.astype(np.float32),
        'pair_i': pair_i.astype(np.int64),
        'pair_j': pair_j.astype(np.int64),
        'pair_label': pair_label.astype(np.float32),
        'pair_confidence': pair_confidence.astype(np.float32),
        'summary_masks': summary_masks.astype(np.float32),
        'teacher_confidence': teacher_confidence.astype(np.float32),
        'meta': dict(meta),
    }


class PreferenceTeacherStore(object):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> Dict[str, Dict]:
        if not path.exists():
            raise FileNotFoundError(f'Preference teacher file not found: {path}')
        obj = np.load(path, allow_pickle=True)
        try:
            obj = obj.item()
        except Exception as exc:
            raise ValueError(f'Invalid preference teacher file format: {path}') from exc
        if not isinstance(obj, dict):
            raise ValueError(
                f'Preference teacher file must contain dict[h5_key -> record], got {type(obj)}'
            )
        if not obj:
            raise ValueError(f'Empty preference teacher file: {path}')
        return obj

    def get(self, h5_key: str) -> Dict:
        if h5_key not in self.records:
            raise KeyError(f'Missing h5 key "{h5_key}" in preference teacher file: {self.path}')
        return validate_preference_record(self.records[h5_key], h5_key=h5_key)

    def keys(self) -> Sequence[str]:
        return tuple(sorted(self.records.keys()))
