from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from helpers.key_helper import canonicalize_video_name, decode_h5_string


class VideoDatasetMIL(object):
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.datasets = self.get_datasets(keys)

    def __getitem__(self, index: int) -> Tuple[
        str, str, np.ndarray, Optional[np.ndarray],
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        key = self.keys[index]
        video_path = Path(key)
        dataset_name = str(video_path.parent)
        video_group_name = video_path.name
        video_file = self.datasets[dataset_name][video_group_name]

        seq = video_file['features'][...].astype(np.float32)

        gtscore = None
        if 'gtscore' in video_file:
            gtscore = video_file['gtscore'][...].astype(np.float32)
            gtscore -= gtscore.min()
            max_value = gtscore.max()
            if max_value > 0:
                gtscore /= max_value

        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)

        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        if 'video_name' in video_file:
            raw_video_name = decode_h5_string(video_file['video_name'][()])
        else:
            raw_video_name = video_group_name

        canonical_video_name = canonicalize_video_name(raw_video_name)

        return (
            key,
            canonical_video_name,
            seq,
            gtscore,
            cps,
            n_frames,
            nfps,
            picks,
            user_summary,
        )

    def __len__(self) -> int:
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]) -> Dict[str, h5py.File]:
        dataset_paths = {str(Path(key).parent) for key in keys}
        return {path: h5py.File(path, 'r') for path in dataset_paths}