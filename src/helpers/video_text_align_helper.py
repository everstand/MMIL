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

# 改进1
def load_rgb_frames_by_picks(
        video_path: PathLike,
        picks: np.ndarray,
        max_consecutive_failures: int = 8,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, int]]:
    """
    Strictly decode RGB frames at the exact absolute frame indices given by `picks`.

    Contracts:
    1. Output frame count must equal len(picks) exactly.
    2. Any decode failure on a requested pick triggers a hard error.
    3. Frames are returned in RGB order.
    4. Picks must be a 1D, non-negative, strictly increasing integer sequence.

    Notes:
    - This implementation does NOT rely on random seek via CAP_PROP_POS_FRAMES.
    - It scans the video sequentially and retrieves frames only when frame_idx hits a target pick.
    - CAP_PROP_FRAME_COUNT is used for audit only, not as a hard truth for control flow.
    """
    video_path = Path(video_path)

    picks_arr = np.asarray(picks)
    if picks_arr.ndim != 1:
        raise ValueError(f'Expected picks to be 1D, got shape {picks_arr.shape}')

    if picks_arr.size == 0:
        raise ValueError('load_rgb_frames_by_picks received empty picks.')

    picks_np = picks_arr.astype(np.int32, copy=False)

    if np.any(picks_np < 0):
        raise ValueError('load_rgb_frames_by_picks received negative picks.')

    if np.any(picks_np[1:] <= picks_np[:-1]):
        raise ValueError(
            'load_rgb_frames_by_picks requires picks to be strictly increasing.'
        )

    cap = cv2.VideoCapture(str(video_path))
    if cap is None or not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path}')

    reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames: List[np.ndarray] = []

    target_ptr = 0
    next_pick = int(picks_np[target_ptr])

    frame_idx = 0
    decode_failures = 0
    consecutive_failures = 0

    while target_ptr < len(picks_np):
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx == next_pick:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                decode_failures += 1
                consecutive_failures += 1
                cap.release()

                raise RuntimeError(
                    f'Failed to decode requested frame {next_pick} '
                    f'from video: {video_path}'
                )

            consecutive_failures = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            target_ptr += 1
            if target_ptr < len(picks_np):
                next_pick = int(picks_np[target_ptr])

        frame_idx += 1

        if consecutive_failures >= max_consecutive_failures:
            cap.release()
            raise RuntimeError(
                f'Too many consecutive decode failures while reading requested picks '
                f'from video: {video_path}'
            )

    cap.release()

    if len(frames) != len(picks_np):
        decoded_count = len(frames)
        first_missing_pick = int(picks_np[decoded_count])

        raise RuntimeError(
            f'Failed to decode all requested picks from video {video_path}. '
            f'decoded={decoded_count}/{len(picks_np)}, '
            f'first_missing_pick={first_missing_pick}, '
            f'last_scanned_frame={frame_idx - 1}'
        )

    audit = {
        'requested_picks': int(len(picks_np)),
        'decoded_frames': int(len(frames)),
        'decode_failures': int(decode_failures),
        'first_pick': int(picks_np[0]),
        'last_pick': int(picks_np[-1]),
        'last_scanned_frame': int(frame_idx - 1),
        'reported_total_frames': int(reported_total_frames),
        'reported_width': int(reported_width),
        'reported_height': int(reported_height),
        'reported_fps': int(reported_fps),
    }

    return picks_np, frames, audit
# end