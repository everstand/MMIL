import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from helpers.mil_path_helper import get_soft_labels_path


logger = logging.getLogger(__name__)


class VideoDatasetMIL(object):
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError('VideoDatasetMIL received empty keys.')

        self.original_keys = list(keys)
        self.datasets = self.get_datasets(self.original_keys)

        self.dataset_names_by_key = {
            key: self.infer_dataset_name_from_key(key) for key in self.original_keys
        }

        self.soft_labels_by_dataset = self.load_soft_labels_by_dataset(
            self.dataset_names_by_key.values()
        )

        self.num_classes_by_dataset = self.validate_soft_labels_by_dataset(
            self.soft_labels_by_dataset
        )

        self.keys = self.filter_valid_keys(
            self.original_keys,
            self.dataset_names_by_key,
            self.soft_labels_by_dataset,
        )

        if not self.keys:
            raise ValueError(
                'No valid training samples remain after soft-label intersection filtering.'
            )

    def __getitem__(self, index: int) -> Tuple[
        str,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        key = self.keys[index]
        video_path = Path(key)
        h5_path = str(video_path.parent)
        h5_key = video_path.name
        video_file = self.datasets[h5_path][h5_key]

        dataset_name = self.dataset_names_by_key[key]
        soft_label = self.soft_labels_by_dataset[dataset_name][h5_key].astype(np.float32)

        seq = video_file['features'][...].astype(np.float32)

        gtscore = None
        if 'gtscore' in video_file:
            gtscore = video_file['gtscore'][...].astype(np.float32)
            gtscore = normalize_gtscore(gtscore)

        user_summary = None
        if 'user_summary' in video_file:
            user_summary = video_file['user_summary'][...].astype(np.float32)

        cps = video_file['change_points'][...].astype(np.int32)
        n_frames = video_file['n_frames'][...].astype(np.int32)
        nfps = video_file['n_frame_per_seg'][...].astype(np.int32)
        picks = video_file['picks'][...].astype(np.int32)

        return (
            key,
            seq,
            soft_label,
            gtscore,
            user_summary,
            cps,
            n_frames,
            nfps,
            picks,
        )

    def __len__(self) -> int:
        return len(self.keys)

    @staticmethod
    def get_datasets(keys: List[str]) -> Dict[str, h5py.File]:
        dataset_paths = {str(Path(key).parent) for key in keys}
        return {path: h5py.File(path, 'r') for path in dataset_paths}

    @staticmethod
    def infer_dataset_name_from_key(key: str) -> str:
        h5_path = str(Path(key).parent).lower()
        if 'tvsum' in h5_path:
            return 'tvsum'
        if 'summe' in h5_path:
            return 'summe'
        raise ValueError(
            f'Cannot infer dataset name from key: {key}. '
            f'Expected HDF5 path to contain "tvsum" or "summe".'
        )

    @staticmethod
    def load_soft_labels_by_dataset(dataset_names) -> Dict[str, Dict[str, np.ndarray]]:
        unique_names = sorted(set(dataset_names))
        soft_labels_by_dataset: Dict[str, Dict[str, np.ndarray]] = {}

        for dataset_name in unique_names:
            soft_label_path = get_soft_labels_path(dataset_name)
            if not soft_label_path.exists():
                raise FileNotFoundError(
                    f'Soft label file not found for dataset "{dataset_name}": {soft_label_path}'
                )

            obj = np.load(soft_label_path, allow_pickle=True)
            try:
                obj = obj.item()
            except Exception as exc:
                raise ValueError(
                    f'Invalid soft label file format for dataset "{dataset_name}": {soft_label_path}'
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f'Soft label file must store a dict[h5_key -> np.ndarray], got {type(obj)} '
                    f'for dataset "{dataset_name}".'
                )

            soft_labels_by_dataset[dataset_name] = obj

        return soft_labels_by_dataset

    @staticmethod
    def validate_soft_labels_by_dataset(
        soft_labels_by_dataset: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, int]:
        num_classes_by_dataset: Dict[str, int] = {}

        for dataset_name, label_dict in soft_labels_by_dataset.items():
            if not label_dict:
                raise ValueError(f'Empty soft label dictionary for dataset "{dataset_name}".')

            expected_dim = None
            for h5_key, value in label_dict.items():
                if not isinstance(value, np.ndarray):
                    raise ValueError(
                        f'Soft label for dataset "{dataset_name}", key "{h5_key}" '
                        f'is not a numpy array: {type(value)}'
                    )
                if value.ndim != 1:
                    raise ValueError(
                        f'Soft label for dataset "{dataset_name}", key "{h5_key}" '
                        f'must be 1D, got shape {value.shape}'
                    )
                if value.size == 0:
                    raise ValueError(
                        f'Soft label for dataset "{dataset_name}", key "{h5_key}" is empty.'
                    )

                if expected_dim is None:
                    expected_dim = int(value.shape[0])
                elif int(value.shape[0]) != expected_dim:
                    raise ValueError(
                        f'Inconsistent soft label dimension in dataset "{dataset_name}": '
                        f'key "{h5_key}" has shape {value.shape}, expected first dim {expected_dim}.'
                    )

            num_classes_by_dataset[dataset_name] = expected_dim

        return num_classes_by_dataset

    @staticmethod
    def filter_valid_keys(
        keys: List[str],
        dataset_names_by_key: Dict[str, str],
        soft_labels_by_dataset: Dict[str, Dict[str, np.ndarray]],
    ) -> List[str]:
        valid_keys: List[str] = []
        filtered_out: Dict[str, List[str]] = {}

        for key in keys:
            h5_key = Path(key).name
            dataset_name = dataset_names_by_key[key]
            label_dict = soft_labels_by_dataset[dataset_name]

            if h5_key in label_dict:
                valid_keys.append(key)
            else:
                filtered_out.setdefault(dataset_name, []).append(key)

        for dataset_name, missing_keys in filtered_out.items():
            logger.warning(
                'Filtered out %d sample(s) from dataset "%s" due to missing soft labels.',
                len(missing_keys),
                dataset_name,
            )

        return valid_keys


def normalize_gtscore(gtscore: np.ndarray) -> np.ndarray:
    gtscore = gtscore.astype(np.float32)
    gtscore = gtscore - gtscore.min()
    max_value = gtscore.max()
    if max_value > 0:
        gtscore = gtscore / max_value
    return gtscore