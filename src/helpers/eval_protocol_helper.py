from typing import Dict

import numpy as np
from scipy.stats import kendalltau, spearmanr


VALID_DATASETS = {'summe', 'tvsum'}


def infer_dataset_name_from_key(key: str) -> str:
    key_lower = str(key).lower()
    if 'tvsum' in key_lower:
        return 'tvsum'
    if 'summe' in key_lower:
        return 'summe'
    raise ValueError(f'Cannot infer dataset name from key: {key}')


def infer_f1_metric_from_dataset(dataset_name: str) -> str:
    dataset_name = dataset_name.strip().lower()
    if dataset_name == 'tvsum':
        return 'avg'
    if dataset_name == 'summe':
        return 'max'
    raise ValueError(f'Unsupported dataset for F1 protocol: {dataset_name}')


def infer_f1_metric_from_key(key: str) -> str:
    return infer_f1_metric_from_dataset(infer_dataset_name_from_key(key))


def validate_1d_same_shape(pred: np.ndarray,
                           target: np.ndarray,
                           key: str,
                           pred_name: str = 'pred',
                           target_name: str = 'target'):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if pred.shape != target.shape:
        raise ValueError(
            f'{pred_name}/{target_name} length mismatch for sample {key}: '
            f'{pred.shape[0]} vs {target.shape[0]}'
        )
    if pred.size < 2:
        raise ValueError(
            f'Need at least 2 elements for rank correlation for sample {key}, got {pred.size}'
        )
    if not np.isfinite(pred).all():
        raise ValueError(f'Non-finite values in {pred_name} for sample {key}')
    if not np.isfinite(target).all():
        raise ValueError(f'Non-finite values in {target_name} for sample {key}')
    return pred, target


def compute_kendall_tau_b(pred: np.ndarray, target: np.ndarray, key: str = '<unknown>') -> float:
    pred, target = validate_1d_same_shape(
        pred, target, key=key, pred_name='pred_scores', target_name='gt_scores'
    )
    result = kendalltau(pred, target, variant='b', nan_policy='propagate')
    return float(result.statistic) if result.statistic is not None else float('nan')


def compute_spearman_rho(pred: np.ndarray, target: np.ndarray, key: str = '<unknown>') -> float:
    pred, target = validate_1d_same_shape(
        pred, target, key=key, pred_name='pred_scores', target_name='gt_scores'
    )
    result = spearmanr(pred, target, nan_policy='propagate')
    return float(result.statistic) if result.statistic is not None else float('nan')


def compute_rank_metrics_from_gtscore(pred_scores: np.ndarray,
                                      gtscore: np.ndarray,
                                      key: str) -> Dict[str, float]:
    """Compute rank metrics against the public-HDF5 aggregated gtscore.

    This is the strict protocol supported by the current DSNet-style public HDF5
    interface used in this repository. It does not claim TVSum per-user score
    averaging unless a per-user score matrix is explicitly added to the dataset.
    """
    pred_scores, gt_scores = validate_1d_same_shape(
        pred_scores,
        gtscore,
        key=key,
        pred_name='summary_scores',
        target_name='gtscore',
    )
    return {
        'kendall': compute_kendall_tau_b(pred_scores, gt_scores, key=key),
        'spearman': compute_spearman_rho(pred_scores, gt_scores, key=key),
    }


def safe_nanmean(values, default: float = 0.0) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float(default)
    return float(np.mean(finite))
