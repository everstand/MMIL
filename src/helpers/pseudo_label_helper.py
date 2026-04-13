import math
from typing import Dict

import numpy as np
import torch


def temporal_window_smooth(frame_features: torch.Tensor,
                           radius: int = 1
                           ) -> torch.Tensor:
    if frame_features.ndim != 2:
        raise ValueError(f'Expected [T, D], got {tuple(frame_features.shape)}')
    if radius < 0:
        raise ValueError(f'Invalid radius: {radius}')

    time_steps, feat_dim = frame_features.shape
    if time_steps == 0:
        raise ValueError('Empty frame feature sequence')

    smoothed = []
    for t in range(time_steps):
        lo = max(0, t - radius)
        hi = min(time_steps, t + radius + 1)
        window_feat = frame_features[lo:hi].mean(dim=0)
        smoothed.append(window_feat)

    return torch.stack(smoothed, dim=0)


def aggregate_soft_labels(frame_text_scores: np.ndarray,
                          top_ratio: float = 0.15
                          ) -> np.ndarray:
    if frame_text_scores.ndim != 2:
        raise ValueError(
            f'Expected frame_text_scores with shape [T, C], got {frame_text_scores.shape}'
        )
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f'Invalid top_ratio: {top_ratio}')

    time_steps, num_classes = frame_text_scores.shape
    top_k = max(1, int(math.ceil(time_steps * top_ratio)))

    soft_labels = np.zeros((num_classes,), dtype=np.float32)
    for class_idx in range(num_classes):
        class_scores = frame_text_scores[:, class_idx]
        sorted_scores = np.sort(class_scores)[::-1]
        soft_labels[class_idx] = sorted_scores[:top_k].mean()

    return soft_labels


def package_pseudo_label_record(frame_text_scores: np.ndarray,
                                soft_labels: np.ndarray
                                ) -> Dict[str, np.ndarray]:
    return {
        'frame_text_scores': frame_text_scores.astype(np.float32),
        'soft_labels': soft_labels.astype(np.float32),
    }