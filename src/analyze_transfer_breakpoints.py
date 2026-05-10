#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose where teacher-to-summary transfer breaks.

This script is offline-only. It may read `gtscore` and `user_summary` for
diagnostics, but it does not construct training labels, update checkpoints, or
change model selection. The goal is to localize the chain:

    teacher shot ranking -> student shot ranking -> knapsack keyshots

Shape contract:
    - model summary scores: [T], aligned to HDF5 `picks`
    - shot scores: [S], aligned to `change_points` and `n_frame_per_seg`
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from anchor_free.dsnet_af_mil_cond import DSNetAFMILCond
from anchor_free.train_mil_cond import (
    aggregate_frame_scores_to_shot_scores,
    build_sampled_to_shot_overlap,
    infer_num_classes,
)
from evaluate_mil_cond import (
    get_keyshot_summ_from_shot_scores,
    project_shot_scores_to_sampled_scores,
)
from modules.shot_context_encoder import (
    build_context_shot_text_stats,
    build_shot_time_features,
)
from helpers import mil_data_helper_cond, vsumm_helper
from helpers.eval_protocol_helper import (
    compute_rank_metrics_from_gtscore,
    infer_f1_metric_from_key,
    safe_nanmean,
)
from helpers.preference_teacher_helper import PreferenceTeacherStore, normalize_01
from helpers.shot_utility_helper import (
    ShotUtilityStore,
    build_budgeted_pseudo_summary_masks,
    resolve_shot_utility_path,
)
from helpers import data_helper
from run_train_mil_cond import load_all_splits, validate_splits


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Offline teacher -> student -> keyshot transfer diagnostic.'
    )
    parser.add_argument('--dataset', type=str, required=True, choices=('summe', 'tvsum'))
    parser.add_argument('--splits', type=str, nargs='+', required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument(
        '--checkpoint-kind',
        type=str,
        default='best_f1',
        choices=('best_f1', 'max_kendall', 'max_spearman'),
    )
    parser.add_argument('--selection-part', type=str, default='val', choices=('train', 'val', 'test', 'all'))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--max-splits', type=int, default=None)

    parser.add_argument(
        '--teacher-kind',
        type=str,
        default='shot_utility',
        choices=('shot_utility', 'preference'),
    )
    parser.add_argument('--preference-teacher-path', type=str, default=None)
    parser.add_argument('--shot-utility-path', type=str, default=None)
    parser.add_argument('--utility-formula', type=str, default='phase1_default')
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--negative-quantile', type=float, default=0.25)
    parser.add_argument('--confidence-threshold', type=float, default=0.6)

    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--base-model', type=str, default='attention', choices=('attention',))
    parser.add_argument('--num-feature', type=int, default=768)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--score-head', type=str, default='dual', choices=('single', 'dual', 'residual_dual'))
    parser.add_argument('--selection-score-source', type=str, default='frame', choices=('frame', 'shot_head'))
    parser.add_argument('--shot-eval-head', type=str, default='selection', choices=('selection', 'rank'))
    parser.add_argument('--shot-head-mode', type=str, default='single', choices=('single', 'dual', 'context'))
    parser.add_argument('--text-cond-num', type=int, default=10)
    parser.add_argument('--caption-coverage-aware', action='store_true')
    parser.add_argument('--text-feature-path', type=str, default=None)
    parser.add_argument('--structured-caption-path', type=str, default=None)

    parser.add_argument('--output-json', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    return parser


def default_split_seed(dataset: str) -> int:
    if dataset == 'summe':
        return 19500
    if dataset == 'tvsum':
        return 12345
    raise ValueError(f'Unsupported dataset: {dataset}')


def selected_keys_for_split(split: Dict, part: str) -> List[Tuple[str, str]]:
    if part == 'train':
        return [('train', key) for key in split['train_keys']]
    if part == 'val':
        return [('val', key) for key in split['val_keys']]
    if part == 'test':
        return [('test', key) for key in split['test_keys']]
    rows = []
    for name in ('train', 'val', 'test'):
        rows.extend((name, key) for key in split[f'{name}_keys'])
    return rows


def checkpoint_path(model_dir: Path, split_idx: int, checkpoint_kind: str) -> Path:
    if checkpoint_kind == 'best_f1':
        name = f'best_model_split{split_idx}.pth'
    elif checkpoint_kind == 'max_kendall':
        name = f'best_model_split{split_idx}_max_kendall.pth'
    elif checkpoint_kind == 'max_spearman':
        name = f'best_model_split{split_idx}_max_spearman.pth'
    else:
        raise ValueError(f'Unsupported checkpoint kind: {checkpoint_kind}')
    path = model_dir / name
    if not path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    return path


def safe_corr(pred: np.ndarray, target: np.ndarray, method: str) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape or pred.size < 2:
        return float('nan')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        return float('nan')
    if float(np.max(pred) - np.min(pred)) <= 1e-12:
        return float('nan')
    if float(np.max(target) - np.min(target)) <= 1e-12:
        return float('nan')
    if method == 'kendall':
        result = kendalltau(pred, target, variant='b', nan_policy='propagate')
    elif method == 'spearman':
        result = spearmanr(pred, target, nan_policy='propagate')
    else:
        raise ValueError(f'Unsupported method: {method}')
    return float(result.statistic) if result.statistic is not None else float('nan')


def sampled_scores_to_shot_scores_exact(sample_scores: np.ndarray,
                                        cps: np.ndarray,
                                        n_frames: int,
                                        picks: np.ndarray) -> np.ndarray:
    sample_scores = np.asarray(sample_scores, dtype=np.float32).reshape(-1)
    picks = np.asarray(picks, dtype=np.int32).reshape(-1)
    cps = np.asarray(cps, dtype=np.int32)
    if sample_scores.shape[0] != picks.shape[0]:
        raise ValueError(f'sample/picks length mismatch: {sample_scores.shape[0]} vs {picks.shape[0]}')
    frame_scores = np.zeros(int(n_frames), dtype=np.float32)
    for idx, pick in enumerate(picks.tolist()):
        pos_hi = int(picks[idx + 1]) if idx + 1 < picks.shape[0] else int(n_frames)
        frame_scores[int(pick):pos_hi] = float(sample_scores[idx])
    shot_scores = np.zeros((cps.shape[0],), dtype=np.float32)
    for shot_idx, (first, last) in enumerate(cps.tolist()):
        shot_scores[shot_idx] = float(frame_scores[int(first):int(last) + 1].mean())
    return shot_scores


def shot_scores_to_sampled_scores(shot_scores: np.ndarray,
                                  cps: np.ndarray,
                                  picks: np.ndarray,
                                  key: str) -> np.ndarray:
    scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    cps = np.asarray(cps, dtype=np.int32)
    picks = np.asarray(picks, dtype=np.int32).reshape(-1)
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Invalid change_points shape for {key}: {cps.shape}')
    if scores.shape[0] != cps.shape[0]:
        raise ValueError(f'shot score length mismatch for {key}: {scores.shape[0]} vs {cps.shape[0]}')
    sample_scores = np.zeros((picks.shape[0],), dtype=np.float32)
    assigned = np.zeros((picks.shape[0],), dtype=bool)
    for shot_idx, (first, last) in enumerate(cps.tolist()):
        mask = (picks >= int(first)) & (picks <= int(last))
        sample_scores[mask] = float(scores[shot_idx])
        assigned[mask] = True
    if not assigned.all():
        missing = np.where(~assigned)[0]
        for idx in missing.tolist():
            frame = int(picks[idx])
            nearest = int(np.argmin(np.minimum(np.abs(cps[:, 0] - frame), np.abs(cps[:, 1] - frame))))
            sample_scores[idx] = float(scores[nearest])
    return sample_scores.astype(np.float32)


def selected_shot_mask_from_scores(shot_scores: np.ndarray,
                                   nfps: np.ndarray,
                                   n_frames: int,
                                   summary_budget: float) -> np.ndarray:
    scores = normalize_01(np.asarray(shot_scores, dtype=np.float32).reshape(-1))
    values = np.round(scores * 1000.0).astype(np.int32)
    weights = np.asarray(nfps, dtype=np.int32).reshape(-1)
    if scores.shape[0] != weights.shape[0]:
        raise ValueError(f'shot score/nfps length mismatch: {scores.shape[0]} vs {weights.shape[0]}')
    capacity = int(int(n_frames) * float(summary_budget))
    if values.size == 0 or int(values.max(initial=0)) <= 0 or capacity <= 0:
        selected_idx = []
    else:
        selected_idx = vsumm_helper.knapsack(values.tolist(), weights.tolist(), capacity)
    mask = np.zeros((scores.shape[0],), dtype=bool)
    if selected_idx:
        mask[np.asarray(selected_idx, dtype=np.int64)] = True
    return mask


def selected_shot_mask_from_raw_scores(shot_scores: np.ndarray,
                                       nfps: np.ndarray,
                                       n_frames: int,
                                       summary_budget: float) -> np.ndarray:
    scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    weights = np.asarray(nfps, dtype=np.int32).reshape(-1)
    if scores.shape[0] != weights.shape[0]:
        raise ValueError(f'shot score/nfps length mismatch: {scores.shape[0]} vs {weights.shape[0]}')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite shot scores in raw-score knapsack mask.')
    capacity = int(int(n_frames) * float(summary_budget))
    values = np.round(np.clip(scores, 0.0, 1.0) * 1000.0).astype(np.int32)
    selected_idx = [] if capacity <= 0 else vsumm_helper.knapsack(values.tolist(), weights.tolist(), capacity)
    mask = np.zeros((scores.shape[0],), dtype=bool)
    if selected_idx:
        mask[np.asarray(selected_idx, dtype=np.int64)] = True
    return mask


def summary_from_selected_shots(selected_mask: np.ndarray,
                                cps: np.ndarray,
                                n_frames: int) -> np.ndarray:
    summary = np.zeros((int(n_frames),), dtype=bool)
    cps = np.asarray(cps, dtype=np.int32)
    for shot_idx in np.where(np.asarray(selected_mask, dtype=bool))[0].tolist():
        first, last = cps[shot_idx]
        first = int(max(0, min(first, int(n_frames) - 1)))
        last = int(max(first, min(last, int(n_frames) - 1)))
        summary[first:last + 1] = True
    return summary


def f1_from_summary(summary: np.ndarray, user_summary: np.ndarray, key: str) -> float:
    return float(
        vsumm_helper.get_summ_f1score(
            pred_summ=summary,
            test_summ=user_summary,
            eval_metric=infer_f1_metric_from_key(key),
        )
    )


def mask_overlap_stats(student_mask: np.ndarray, teacher_mask: np.ndarray) -> Dict[str, float]:
    student = np.asarray(student_mask, dtype=bool).reshape(-1)
    teacher = np.asarray(teacher_mask, dtype=bool).reshape(-1)
    if student.shape != teacher.shape:
        raise ValueError(f'mask length mismatch: {student.shape} vs {teacher.shape}')
    inter = float(np.logical_and(student, teacher).sum())
    union = float(np.logical_or(student, teacher).sum())
    student_count = float(student.sum())
    teacher_count = float(teacher.sum())
    return {
        'selected_jaccard': inter / union if union > 0 else 0.0,
        'selected_precision_vs_teacher': inter / student_count if student_count > 0 else 0.0,
        'selected_recall_vs_teacher': inter / teacher_count if teacher_count > 0 else 0.0,
        'student_selected_shots': student_count,
        'teacher_selected_shots': teacher_count,
    }


def preference_pair_accuracy(student_shot_scores: np.ndarray,
                             record: Dict) -> float:
    pair_i = np.asarray(record['pair_i'], dtype=np.int64).reshape(-1)
    pair_j = np.asarray(record['pair_j'], dtype=np.int64).reshape(-1)
    pair_conf = np.asarray(record['pair_confidence'], dtype=np.float32).reshape(-1)
    if pair_i.size == 0:
        return float('nan')
    scores = np.asarray(student_shot_scores, dtype=np.float32).reshape(-1)
    correct = scores[pair_i] > scores[pair_j]
    if pair_conf.shape[0] != pair_i.shape[0]:
        return float(correct.mean())
    weight = np.clip(pair_conf, 0.0, 1.0)
    denom = float(weight.sum())
    if denom <= 1e-8:
        return float(correct.mean())
    return float((correct.astype(np.float32) * weight).sum() / denom)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def bce_np(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError('BCE arrays must have matching shape.')
    if not mask.any():
        return 0.0
    pred = np.clip(pred[mask], 1e-6, 1.0 - 1e-6)
    target = target[mask]
    return float(-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred)).mean())


def budget_ratio_from_shot_scores(shot_scores: np.ndarray, nfps: np.ndarray, n_frames: int) -> float:
    scores = np.asarray(shot_scores, dtype=np.float32).reshape(-1)
    weights = np.asarray(nfps, dtype=np.float32).reshape(-1)
    if scores.shape != weights.shape:
        raise ValueError(f'budget score/nfps mismatch: {scores.shape} vs {weights.shape}')
    return float(np.sum(scores * weights) / max(float(n_frames), 1.0))


def load_teacher_scores(args, h5_key: str, cps: np.ndarray, nfps: np.ndarray, n_frames: int):
    if args.teacher_kind == 'preference':
        store = load_teacher_scores.preference_store
        record = store.get(h5_key)
        teacher_scores = np.asarray(record['shot_scores'], dtype=np.float32).reshape(-1)
        target = np.asarray(record['inclusion_prob'], dtype=np.float32).reshape(-1)
        confidence = np.asarray(record['teacher_confidence'], dtype=np.float32).reshape(-1)
        supervised = confidence >= float(args.confidence_threshold)
        selected_mask = selected_shot_mask_from_scores(
            teacher_scores, nfps=nfps, n_frames=n_frames, summary_budget=args.summary_budget
        )
        meta = {
            'teacher_num_pairs': int(np.asarray(record['pair_i']).reshape(-1).shape[0]),
            'teacher_num_positive': int((target >= 0.60).sum()),
            'teacher_num_negative': int((target <= 0.20).sum()),
            'teacher_pair_accuracy': None,
            'teacher_bce_on_student': None,
            'teacher_budget_gap_on_student': None,
            'record': record,
            'target': target,
            'supervised': supervised,
        }
        return normalize_01(teacher_scores), selected_mask, meta

    store = load_teacher_scores.utility_store
    teacher_scores = np.asarray(
        store.get(h5_key=h5_key, formula_name=args.utility_formula), dtype=np.float32
    ).reshape(-1)
    masks = build_budgeted_pseudo_summary_masks(
        utility=teacher_scores,
        cps=cps,
        nfps=nfps,
        n_frames=n_frames,
        summary_budget=args.summary_budget,
        negative_quantile=args.negative_quantile,
    )
    meta = {
        'teacher_num_pairs': 0,
        'teacher_num_positive': int(np.asarray(masks['selected_mask'], dtype=bool).sum()),
        'teacher_num_negative': int(np.asarray(masks['negative_mask'], dtype=bool).sum()),
        'teacher_pair_accuracy': None,
        'teacher_bce_on_student': None,
        'teacher_budget_gap_on_student': None,
        'target': np.asarray(masks['target'], dtype=np.float32),
        'supervised': np.asarray(masks['supervised_mask'], dtype=bool),
        'record': None,
    }
    return normalize_01(teacher_scores), np.asarray(masks['selected_mask'], dtype=bool), meta


load_teacher_scores.preference_store = None
load_teacher_scores.utility_store = None


def predict_one(model, sample, args):
    (
        key,
        seq,
        _soft_label,
        text_cond,
        _text_target,
        _all_text_features,
        _caption_spans_idx,
        _caption_valid_mask,
        gtscore,
        user_summary,
        cps,
        n_frames,
        nfps,
        picks,
        text_cond_mask,
        caption_coverage_ratio,
    ) = sample

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
    text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(args.device)
    text_cond_mask_tensor = torch.tensor(text_cond_mask, dtype=torch.float32, device=args.device)

    picks_np = np.asarray(picks, dtype=np.int32).reshape(-1)
    cps_np = np.asarray(cps, dtype=np.int32)
    nfps_np = np.asarray(nfps, dtype=np.int32).reshape(-1)
    n_frames_int = int(np.asarray(n_frames).item())

    overlaps, shot_lengths = build_sampled_to_shot_overlap(
        picks=torch.tensor(picks_np, dtype=torch.long, device=args.device),
        cps=torch.tensor(cps_np, dtype=torch.long, device=args.device),
        n_frames=n_frames_int,
    )

    with torch.no_grad():
        if args.selection_score_source == 'shot_head':
            (
                _instance_logits,
                _pool_logits,
                _frame_summary_scores,
                _bag_logits,
                _summary_feat,
                _pre_cross_frame_repr,
                cond_frame_repr,
                _hidden_frame_repr,
            ) = model(
                seq_tensor,
                text_cond_tensor,
                text_cond_mask_tensor if args.caption_coverage_aware else None,
            )
            if args.shot_head_mode == 'context':
                all_text_features_tensor = torch.tensor(
                    _all_text_features, dtype=torch.float32, device=args.device
                )
                caption_spans_idx_tensor = torch.tensor(
                    _caption_spans_idx, dtype=torch.long, device=args.device
                )
                caption_valid_mask_tensor = torch.tensor(
                    _caption_valid_mask, dtype=torch.float32, device=args.device
                )
                shot_text_feat, _shot_mass_density, _valid_shots = build_context_shot_text_stats(
                    caption_spans_idx=caption_spans_idx_tensor,
                    caption_valid_mask=caption_valid_mask_tensor,
                    all_text_features=all_text_features_tensor,
                    overlaps=overlaps,
                    shot_lengths=shot_lengths,
                )
                shot_time_feat = build_shot_time_features(
                    cps=torch.tensor(cps_np, dtype=torch.long, device=args.device),
                    nfps=torch.tensor(nfps_np, dtype=torch.float32, device=args.device),
                    n_frames=n_frames_int,
                )
            else:
                shot_text_feat = None
                shot_time_feat = None
            shot_scores_tensor = model.predict_shot_scores(
                frame_repr=cond_frame_repr,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
                head=args.shot_eval_head,
                shot_text_feat=shot_text_feat,
                shot_time_feat=shot_time_feat,
            )
            selection_shot_scores = shot_scores_tensor.detach().cpu().numpy().astype(np.float32)
            summary_scores = project_shot_scores_to_sampled_scores(
                shot_scores=shot_scores_tensor,
                overlaps=overlaps,
            ).detach().cpu().numpy().astype(np.float32)
            exact_shot_scores = selection_shot_scores
        else:
            summary_scores = model.predict_summary_scores(
                seq_tensor,
                text_cond_tensor,
                text_cond_mask_tensor if args.caption_coverage_aware else None,
            ).detach().cpu().numpy().astype(np.float32)
            selection_shot_scores = aggregate_frame_scores_to_shot_scores(
                frame_scores=torch.tensor(summary_scores, dtype=torch.float32, device=args.device),
                overlaps=overlaps,
                shot_lengths=shot_lengths,
            ).detach().cpu().numpy().astype(np.float32)
            exact_shot_scores = sampled_scores_to_shot_scores_exact(
                summary_scores,
                cps=cps_np,
                n_frames=n_frames_int,
                picks=picks_np,
            )

    if summary_scores.shape[0] != picks_np.shape[0]:
        raise ValueError(
            f'Summary score length mismatch for {key}: '
            f'{summary_scores.shape[0]} vs picks {picks_np.shape[0]}'
        )
    human_shot_scores = aggregate_frame_scores_to_shot_scores(
        frame_scores=torch.tensor(np.asarray(gtscore, dtype=np.float32), dtype=torch.float32, device=args.device),
        overlaps=overlaps,
        shot_lengths=shot_lengths,
    ).detach().cpu().numpy().astype(np.float32)

    if args.selection_score_source == 'shot_head':
        pred_summary = get_keyshot_summ_from_shot_scores(
            shot_scores=selection_shot_scores,
            cps=cps_np,
            nfps=nfps_np,
            n_frames=n_frames_int,
            summary_budget=args.summary_budget,
        )
    else:
        pred_summary = vsumm_helper.get_keyshot_summ(
            summary_scores,
            cps_np,
            n_frames_int,
            nfps_np,
            picks_np,
            proportion=args.summary_budget,
        )
    if args.selection_score_source == 'shot_head':
        student_selected_mask = selected_shot_mask_from_raw_scores(
            selection_shot_scores,
            nfps=nfps_np,
            n_frames=n_frames_int,
            summary_budget=args.summary_budget,
        )
    else:
        student_selected_mask = selected_shot_mask_from_scores(
            exact_shot_scores,
            nfps=nfps_np,
            n_frames=n_frames_int,
            summary_budget=args.summary_budget,
        )
    return {
        'key': str(key),
        'h5_key': Path(str(key)).name,
        'summary_scores': summary_scores,
        'selection_shot_scores': selection_shot_scores,
        'exact_shot_scores': exact_shot_scores,
        'human_shot_scores': human_shot_scores,
        'gtscore': np.asarray(gtscore, dtype=np.float32).reshape(-1),
        'user_summary': np.asarray(user_summary, dtype=np.float32),
        'cps': cps_np,
        'nfps': nfps_np,
        'n_frames': n_frames_int,
        'picks': picks_np,
        'pred_summary': pred_summary,
        'student_selected_mask': student_selected_mask,
        'caption_coverage_ratio': float(np.asarray(caption_coverage_ratio).item()),
    }


def analyze_sample(pred, args) -> Dict[str, float]:
    h5_key = pred['h5_key']
    teacher_scores, teacher_selected_mask, teacher_meta = load_teacher_scores(
        args,
        h5_key=h5_key,
        cps=pred['cps'],
        nfps=pred['nfps'],
        n_frames=pred['n_frames'],
    )
    if teacher_scores.shape[0] != pred['exact_shot_scores'].shape[0]:
        raise ValueError(
            f'Teacher/student shot length mismatch for {h5_key}: '
            f'{teacher_scores.shape[0]} vs {pred["exact_shot_scores"].shape[0]}'
        )

    teacher_sample_scores = shot_scores_to_sampled_scores(
        teacher_scores,
        cps=pred['cps'],
        picks=pred['picks'],
        key=pred['key'],
    )
    teacher_summary = get_keyshot_summ_from_shot_scores(
        shot_scores=teacher_scores,
        cps=pred['cps'],
        nfps=pred['nfps'],
        n_frames=pred['n_frames'],
        summary_budget=args.summary_budget,
    )

    student_rank = compute_rank_metrics_from_gtscore(
        pred_scores=pred['summary_scores'],
        gtscore=pred['gtscore'],
        key=pred['key'],
    )
    teacher_rank = compute_rank_metrics_from_gtscore(
        pred_scores=teacher_sample_scores,
        gtscore=pred['gtscore'],
        key=pred['key'],
    )

    student_f1 = f1_from_summary(pred['pred_summary'], pred['user_summary'], pred['key'])
    teacher_f1 = f1_from_summary(teacher_summary, pred['user_summary'], pred['key'])

    overlap = mask_overlap_stats(pred['student_selected_mask'], teacher_selected_mask)
    student_vs_teacher_tau = safe_corr(pred['exact_shot_scores'], teacher_scores, 'kendall')
    student_vs_teacher_rho = safe_corr(pred['exact_shot_scores'], teacher_scores, 'spearman')
    student_shot_tau = safe_corr(pred['exact_shot_scores'], pred['human_shot_scores'], 'kendall')
    student_shot_rho = safe_corr(pred['exact_shot_scores'], pred['human_shot_scores'], 'spearman')
    teacher_shot_tau = safe_corr(teacher_scores, pred['human_shot_scores'], 'kendall')
    teacher_shot_rho = safe_corr(teacher_scores, pred['human_shot_scores'], 'spearman')

    if args.teacher_kind == 'preference':
        teacher_meta['teacher_pair_accuracy'] = preference_pair_accuracy(
            pred['exact_shot_scores'],
            teacher_meta['record'],
        )
        target = np.asarray(teacher_meta['target'], dtype=np.float32)
        supervised = np.asarray(teacher_meta['supervised'], dtype=bool)
        teacher_meta['teacher_bce_on_student'] = bce_np(
            pred=np.clip(pred['exact_shot_scores'], 0.0, 1.0),
            target=target,
            mask=supervised,
        )
    else:
        target = np.asarray(teacher_meta['target'], dtype=np.float32)
        supervised = np.asarray(teacher_meta['supervised'], dtype=bool)
        teacher_meta['teacher_bce_on_student'] = bce_np(
            pred=np.clip(pred['exact_shot_scores'], 0.0, 1.0),
            target=target,
            mask=supervised,
        )

    student_budget_ratio_soft = budget_ratio_from_shot_scores(
        pred['exact_shot_scores'], nfps=pred['nfps'], n_frames=pred['n_frames']
    )
    teacher_meta['teacher_budget_gap_on_student'] = abs(
        student_budget_ratio_soft - float(args.summary_budget)
    )

    row = {
        'h5_key': h5_key,
        'teacher_F1': float(teacher_f1),
        'teacher_Tau_frame': float(teacher_rank['kendall']),
        'teacher_Rho_frame': float(teacher_rank['spearman']),
        'teacher_Tau_shot': float(teacher_shot_tau),
        'teacher_Rho_shot': float(teacher_shot_rho),
        'student_F1': float(student_f1),
        'student_Tau_frame': float(student_rank['kendall']),
        'student_Rho_frame': float(student_rank['spearman']),
        'student_Tau_shot': float(student_shot_tau),
        'student_Rho_shot': float(student_shot_rho),
        'student_teacher_Tau_shot': float(student_vs_teacher_tau),
        'student_teacher_Rho_shot': float(student_vs_teacher_rho),
        'student_minus_teacher_F1': float(student_f1 - teacher_f1),
        'student_minus_teacher_Tau_frame': float(student_rank['kendall'] - teacher_rank['kendall']),
        'student_minus_teacher_Rho_frame': float(student_rank['spearman'] - teacher_rank['spearman']),
        'teacher_num_pairs': int(teacher_meta['teacher_num_pairs']),
        'teacher_num_positive': int(teacher_meta['teacher_num_positive']),
        'teacher_num_negative': int(teacher_meta['teacher_num_negative']),
        'teacher_pair_accuracy': (
            float(teacher_meta['teacher_pair_accuracy'])
            if teacher_meta['teacher_pair_accuracy'] is not None
            else float('nan')
        ),
        'teacher_bce_on_student': float(teacher_meta['teacher_bce_on_student']),
        'student_soft_budget_ratio': float(student_budget_ratio_soft),
        'student_budget_gap_soft': float(teacher_meta['teacher_budget_gap_on_student']),
        'student_keyshot_budget_ratio': float(np.asarray(pred['pred_summary'], dtype=np.float32).sum() / max(float(pred['n_frames']), 1.0)),
        'teacher_keyshot_budget_ratio': float(np.asarray(teacher_summary, dtype=np.float32).sum() / max(float(pred['n_frames']), 1.0)),
        'caption_coverage_ratio': float(pred['caption_coverage_ratio']),
    }
    row.update(overlap)
    return row


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(rows: List[Dict]) -> Dict:
    numeric_keys = sorted(
        key for key in {k for row in rows for k in row.keys()}
        if key not in ('key', 'h5_key', 'part', 'checkpoint_path')
    )
    summary = {'num_rows': int(len(rows))}
    for key in numeric_keys:
        vals = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                vals.append(float(value))
        if vals:
            mean, std = mean_std(vals)
            summary[f'{key}_mean'] = mean
            summary[f'{key}_std'] = std

    teacher_ok_student_teacher_bad = [
        row for row in rows
        if row['teacher_F1'] >= 0.45
        and row['teacher_Tau_frame'] >= 0.08
        and (not np.isfinite(row['student_teacher_Tau_shot']) or row['student_teacher_Tau_shot'] < 0.05)
    ]
    student_rank_ok_summary_bad = [
        row for row in rows
        if row['student_Tau_frame'] >= 0.08 and row['student_F1'] < 0.40
    ]
    teacher_beats_student_rank = [
        row for row in rows
        if row['teacher_Tau_frame'] > row['student_Tau_frame']
    ]
    student_beats_teacher_f1 = [
        row for row in rows
        if row['student_F1'] > row['teacher_F1']
    ]
    summary.update({
        'teacher_ok_student_teacher_bad_count': int(len(teacher_ok_student_teacher_bad)),
        'student_rank_ok_summary_bad_count': int(len(student_rank_ok_summary_bad)),
        'teacher_beats_student_rank_count': int(len(teacher_beats_student_rank)),
        'student_beats_teacher_f1_count': int(len(student_beats_teacher_f1)),
    })
    return summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_model_for_split(args, train_keys: List[str]) -> DSNetAFMILCond:
    train_set = mil_data_helper_cond.VideoDatasetMILCond(
        train_keys,
        text_cond_num=args.text_cond_num,
        random_text_sampling=False,
        caption_coverage_aware=args.caption_coverage_aware,
        text_feature_path=args.text_feature_path,
        structured_caption_path=args.structured_caption_path,
    )
    num_classes = infer_num_classes(train_set)
    return DSNetAFMILCond(
        base_model=args.base_model,
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        num_classes=num_classes,
        score_head=args.score_head,
        use_shot_head=args.selection_score_source == 'shot_head',
        shot_head_mode=args.shot_head_mode,
    ).to(args.device)


def main() -> None:
    args = get_parser().parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is unavailable.')

    if args.selection_score_source == 'shot_head' and args.shot_eval_head == 'rank' and args.shot_head_mode not in ('dual', 'context'):
        raise ValueError('--shot-eval-head rank requires --shot-head-mode dual or context.')

    effective_seed = int(args.seed) if args.seed is not None else default_split_seed(args.dataset)
    splits = load_all_splits(args.splits, val_ratio=args.val_ratio, seed=effective_seed)
    if args.max_splits is not None:
        if args.max_splits <= 0:
            raise ValueError(f'Invalid max_splits={args.max_splits}')
        splits = splits[:args.max_splits]
    validate_splits(splits, args.dataset)

    if args.teacher_kind == 'preference':
        if not args.preference_teacher_path:
            raise ValueError('--teacher-kind preference requires --preference-teacher-path.')
        load_teacher_scores.preference_store = PreferenceTeacherStore(Path(args.preference_teacher_path))
    else:
        utility_path = resolve_shot_utility_path(
            dataset_name=args.dataset,
            explicit_path=args.shot_utility_path,
        )
        load_teacher_scores.utility_store = ShotUtilityStore(utility_path)

    rows: List[Dict] = []
    model_dir = Path(args.model_dir)
    for split_idx, split in enumerate(splits):
        ckpt = checkpoint_path(model_dir, split_idx, args.checkpoint_kind)
        model = build_model_for_split(args, split['train_keys'])
        state_dict = torch.load(str(ckpt), map_location=args.device)
        model.load_state_dict(state_dict)
        model.eval()

        selected = selected_keys_for_split(split, args.selection_part)
        part_to_keys: Dict[str, List[str]] = {}
        for part, key in selected:
            part_to_keys.setdefault(part, []).append(key)

        for part, keys in part_to_keys.items():
            dataset = mil_data_helper_cond.VideoDatasetMILCond(
                keys,
                text_cond_num=args.text_cond_num,
                random_text_sampling=False,
                caption_coverage_aware=args.caption_coverage_aware,
                text_feature_path=args.text_feature_path,
                structured_caption_path=args.structured_caption_path,
            )
            loader = data_helper.DataLoader(dataset, shuffle=False)
            for sample in loader:
                pred = predict_one(model, sample, args)
                row = analyze_sample(pred, args)
                row.update({
                    'fold_idx': int(split_idx),
                    'part': part,
                    'key': pred['key'],
                    'checkpoint_path': str(ckpt),
                })
                rows.append(row)

    result = {
        'dataset': args.dataset,
        'splits': list(args.splits),
        'seed': int(effective_seed),
        'seed_explicit': bool(args.seed is not None),
        'val_ratio': float(args.val_ratio),
        'model_dir': str(model_dir),
        'checkpoint_kind': args.checkpoint_kind,
        'selection_part': args.selection_part,
        'teacher_kind': args.teacher_kind,
        'preference_teacher_path': args.preference_teacher_path,
        'shot_utility_path': args.shot_utility_path,
        'utility_formula': args.utility_formula,
        'summary_budget': float(args.summary_budget),
        'selection_score_source': args.selection_score_source,
        'shot_eval_head': args.shot_eval_head,
        'shot_head_mode': args.shot_head_mode,
        'summary': summarize(rows),
        'rows': rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=True)
    write_csv(Path(args.output_csv), rows)

    print(json.dumps(result['summary'], indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == '__main__':
    main()
