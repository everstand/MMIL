from os import PathLike
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_sampled_rgb_frames(video_path: PathLike,
                            sample_rate: int
                            ) -> Tuple[int, np.ndarray, List[np.ndarray]]:
    n_frames, picks, frames, _ = load_sampled_rgb_frames_with_audit(
        video_path=video_path,
        sample_rate=sample_rate,
    )
    return n_frames, picks, frames


def load_sampled_rgb_frames_with_audit(
        video_path: PathLike,
        sample_rate: int,
        max_consecutive_failures: int = 64
) -> Tuple[int, np.ndarray, List[np.ndarray], Dict[str, int]]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if cap is None or not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path}')

    frames: List[np.ndarray] = []
    picks: List[int] = []

    frame_idx = 0
    decode_failures = 0
    consecutive_failures = 0

    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % sample_rate == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                decode_failures += 1
                consecutive_failures += 1
                frame_idx += 1

                if consecutive_failures >= max_consecutive_failures:
                    break
                continue

            consecutive_failures = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            picks.append(frame_idx)

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f'No valid sampled frames decoded from video: {video_path}')

    picks_np = np.asarray(picks, dtype=np.int32)
    expected_sampled_frames = (frame_idx + sample_rate - 1) // sample_rate

    audit = {
        'n_frames': int(frame_idx),
        'decode_failures': int(decode_failures),
        'expected_sampled_frames': int(expected_sampled_frames),
        'valid_sampled_frames': int(len(frames)),
        'sample_rate': int(sample_rate),
    }
    return frame_idx, picks_np, frames, audit