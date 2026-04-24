import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from helpers.mil_path_helper import (
    get_openclip_feature_store_path,
    get_soft_labels_path,
    get_structured_caption_json_path,
    get_text_feature_store_path,
)


logger = logging.getLogger(__name__)


class VideoDatasetMILCond(object):
    def __init__(self,
                 keys: List[str],
                 text_cond_num: int = 7,
                 random_text_sampling: bool = True):
        if not keys:
            raise ValueError('VideoDatasetMILCond received empty keys.')
        if text_cond_num <= 0:
            raise ValueError(f'Invalid text_cond_num: {text_cond_num}')

        self.original_keys = list(keys)
        self.text_cond_num = int(text_cond_num)
        self.random_text_sampling = bool(random_text_sampling)

        self.h5_datasets = self.get_h5_datasets(self.original_keys)

        self.dataset_names_by_key = {
            key: self.infer_dataset_name_from_key(key) for key in self.original_keys
        }

        self.soft_labels_by_dataset = self.load_soft_labels_by_dataset(
            self.dataset_names_by_key.values()
        )
        self.num_classes_by_dataset = self.validate_soft_labels_by_dataset(
            self.soft_labels_by_dataset
        )

        self.visual_feature_stores = self.load_visual_feature_stores(
            self.dataset_names_by_key.values()
        )
        self.text_feature_stores = self.load_text_feature_stores(
            self.dataset_names_by_key.values()
        )
        self.structured_captions_by_dataset = self.load_structured_captions_by_dataset(
            self.dataset_names_by_key.values()
        )

        self.keys = self.validate_and_filter_keys(
            self.original_keys,
            self.dataset_names_by_key,
            self.soft_labels_by_dataset,
            self.visual_feature_stores,
            self.text_feature_stores,
            self.structured_captions_by_dataset,
        )

        if not self.keys:
            raise ValueError(
                'No valid samples remain after conditioned data validation.'
            )

    def __getitem__(self, index: int):
        key = self.keys[index]
        key_path = Path(key)
        h5_path = str(key_path.parent)
        h5_key = key_path.name

        dataset_name = self.dataset_names_by_key[key]

        h5_group = self.h5_datasets[h5_path][h5_key]
        visual_group = self.visual_feature_stores[dataset_name][h5_key]
        text_group = self.text_feature_stores[dataset_name][h5_key]
        structured_entry = self.structured_captions_by_dataset[dataset_name][h5_key]

        seq = visual_group['features'][...].astype(np.float32)
        if seq.ndim != 2:
            raise ValueError(f'Expected seq shape [T, D], got {seq.shape} for {key}')

        all_text_features = text_group['all_text_features'][...].astype(np.float32)
        text_target = text_group['text_target'][...].astype(np.float32)

        if all_text_features.ndim != 2:
            raise ValueError(
                f'Expected all_text_features shape [N, D], got {all_text_features.shape} for {key}'
            )
        if text_target.ndim != 1:
            raise ValueError(
                f'Expected text_target shape [D], got {text_target.shape} for {key}'
            )
        if all_text_features.shape[1] != text_target.shape[0]:
            raise ValueError(
                f'Text feature dim mismatch for {key}: '
                f'all_text_features={all_text_features.shape}, text_target={text_target.shape}'
            )
        
        if seq.shape[1] != text_target.shape[0]:
            raise ValueError(
                f'Visual/text feature dim mismatch for {key}: '
                f'seq_dim={seq.shape[1]} vs text_dim={text_target.shape[0]}'
            )

        captions = extract_caption_items(structured_entry)
        if len(captions) != all_text_features.shape[0]:
            raise ValueError(
                f'Caption count mismatch for {key}: '
                f'structured={len(captions)} vs text_features={all_text_features.shape[0]}'
            )

        fps = extract_video_fps(structured_entry, h5_key)

        text_cond = self.sample_text_cond(all_text_features)

        soft_label = self.soft_labels_by_dataset[dataset_name][h5_key].astype(np.float32)

        gtscore = None
        if 'gtscore' in h5_group:
            gtscore = h5_group['gtscore'][...].astype(np.float32)
            gtscore = normalize_gtscore(gtscore)

        user_summary = None
        if 'user_summary' in h5_group:
            user_summary = h5_group['user_summary'][...].astype(np.float32)

        cps = h5_group['change_points'][...].astype(np.int32)
        n_frames = h5_group['n_frames'][...].astype(np.int32)
        nfps = h5_group['n_frame_per_seg'][...].astype(np.int32)
        picks = h5_group['picks'][...].astype(np.int32)

        if picks.shape[0] != seq.shape[0]:
            raise ValueError(
                f'Visual feature length mismatch for {key}: '
                f'len(picks)={picks.shape[0]} vs seq_len={seq.shape[0]}'
            )

        caption_spans_idx, caption_valid_mask = build_caption_spans_idx(
            captions=captions,
            picks=picks,
            fps=fps,
        )

        return (
            key,
            seq,
            soft_label,
            text_cond,
            text_target,
            all_text_features,
            caption_spans_idx,
            caption_valid_mask,
            gtscore,
            user_summary,
            cps,
            n_frames,
            nfps,
            picks,
        )

    def __len__(self) -> int:
        return len(self.keys)

    def sample_text_cond(self, all_text_features: np.ndarray) -> np.ndarray:
        num_captions = int(all_text_features.shape[0])

        if num_captions <= 0:
            raise ValueError('Cannot sample text_cond from empty all_text_features.')

        if num_captions >= self.text_cond_num:
            if self.random_text_sampling:
                indices = sorted(random.sample(range(num_captions), self.text_cond_num))
            else:
                if self.text_cond_num == 1:
                    indices = [0]
                else:
                    indices = np.linspace(
                        0,
                        num_captions - 1,
                        num=self.text_cond_num,
                    ).round().astype(np.int32).tolist()
        else:
            # Low-pace videos may legitimately have fewer visual segments than text_cond_num.
            # Repeat available caption features to keep the model input shape fixed.
            if self.text_cond_num == 1:
                indices = [0]
            else:
                indices = np.linspace(
                    0,
                    num_captions - 1,
                    num=self.text_cond_num,
                ).round().astype(np.int32).tolist()

        text_cond = all_text_features[indices].astype(np.float32)
        if text_cond.shape != (self.text_cond_num, all_text_features.shape[1]):
            raise ValueError(
                f'Invalid text_cond shape after sampling: {text_cond.shape}'
            )
        return text_cond

    @staticmethod
    def get_h5_datasets(keys: List[str]) -> Dict[str, h5py.File]:
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
    def load_visual_feature_stores(dataset_names) -> Dict[str, h5py.File]:
        unique_names = sorted(set(dataset_names))
        visual_stores: Dict[str, h5py.File] = {}

        for dataset_name in unique_names:
            store_path = get_openclip_feature_store_path(dataset_name)
            if not store_path.exists():
                raise FileNotFoundError(
                    f'OpenCLIP visual feature store not found for dataset "{dataset_name}": {store_path}'
                )
            visual_stores[dataset_name] = h5py.File(store_path, 'r')

        return visual_stores

    @staticmethod
    def load_text_feature_stores(dataset_names) -> Dict[str, h5py.File]:
        unique_names = sorted(set(dataset_names))
        text_stores: Dict[str, h5py.File] = {}

        for dataset_name in unique_names:
            store_path = get_text_feature_store_path(dataset_name)
            if not store_path.exists():
                raise FileNotFoundError(
                    f'Text feature store not found for dataset "{dataset_name}": {store_path}'
                )
            text_stores[dataset_name] = h5py.File(store_path, 'r')

        return text_stores

    @staticmethod
    def load_structured_captions_by_dataset(dataset_names) -> Dict[str, Dict]:
        unique_names = sorted(set(dataset_names))
        structured_by_dataset: Dict[str, Dict] = {}

        for dataset_name in unique_names:
            json_path = get_structured_caption_json_path(dataset_name)
            if not json_path.exists():
                raise FileNotFoundError(
                    f'Structured caption json not found for dataset "{dataset_name}": {json_path}'
                )

            with open(json_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)

            if not isinstance(obj, dict):
                raise ValueError(
                    f'Structured caption json must store dict[h5_key -> entry], got {type(obj)} '
                    f'for dataset "{dataset_name}".'
                )

            structured_by_dataset[dataset_name] = obj

        return structured_by_dataset

    @staticmethod
    def validate_and_filter_keys(
        keys: List[str],
        dataset_names_by_key: Dict[str, str],
        soft_labels_by_dataset: Dict[str, Dict[str, np.ndarray]],
        visual_feature_stores: Dict[str, h5py.File],
        text_feature_stores: Dict[str, h5py.File],
        structured_captions_by_dataset: Dict[str, Dict],
    ) -> List[str]:
        """Validate key coverage for all required conditioned-training assets.

        Formal SumMe/TVSum experiments must be fail-fast: no split key may be
        silently removed because a pseudo label, visual feature, text feature,
        or structured caption is missing.
        """
        valid_keys: List[str] = []

        for key in keys:
            h5_key = Path(key).name
            dataset_name = dataset_names_by_key[key]

            label_dict = soft_labels_by_dataset[dataset_name]
            if h5_key not in label_dict:
                raise ValueError(
                    f'Missing soft label for key "{h5_key}" in dataset "{dataset_name}". '
                    'Do not silently filter formal split samples.'
                )

            if h5_key not in visual_feature_stores[dataset_name]:
                raise ValueError(
                    f'Missing visual feature group for key "{h5_key}" in dataset "{dataset_name}".'
                )

            if h5_key not in text_feature_stores[dataset_name]:
                raise ValueError(
                    f'Missing text feature group for key "{h5_key}" in dataset "{dataset_name}".'
                )

            if h5_key not in structured_captions_by_dataset[dataset_name]:
                raise ValueError(
                    f'Missing structured caption entry for key "{h5_key}" in dataset "{dataset_name}".'
                )

            valid_keys.append(key)

        return valid_keys

def extract_caption_items(structured_entry: Dict) -> List[Dict]:
    if not isinstance(structured_entry, dict):
        raise ValueError(f'Invalid structured caption entry type: {type(structured_entry)}')

    captions = structured_entry.get('captions', None)
    if not isinstance(captions, list) or not captions:
        raise ValueError('Structured caption entry must contain a non-empty "captions" list.')
    return captions


def extract_video_fps(structured_entry: Dict, h5_key: str) -> float:
    sample_meta = structured_entry.get('sample_meta', None)
    if not isinstance(sample_meta, dict):
        raise ValueError(f'Missing or invalid sample_meta for {h5_key}')

    fps = sample_meta.get('fps', None)
    if fps is None:
        raise ValueError(f'Missing fps in structured caption sample_meta for {h5_key}')

    fps = float(fps)
    if fps <= 0:
        raise ValueError(f'Invalid fps={fps} for {h5_key}')

    return fps


def parse_mmss(ts: str) -> float:
    if not isinstance(ts, str) or ':' not in ts:
        raise ValueError(f'Invalid MM:SS timestamp: {ts}')
    mm, ss = ts.split(':')
    return int(mm) * 60.0 + int(ss)


def build_caption_spans_idx(captions: List[Dict],
                            picks: np.ndarray,
                            fps: float) -> Tuple[np.ndarray, np.ndarray]:
    time_axis_sec = picks.astype(np.float32) / float(fps)

    num_captions = len(captions)
    spans = np.zeros((num_captions, 2), dtype=np.int32)
    valid = np.zeros((num_captions,), dtype=np.float32)

    min_duration = 1.0 / float(fps)

    for k, item in enumerate(captions):
        start_sec = parse_mmss(item['start_mmss'])
        end_sec = parse_mmss(item['end_mmss'])
        if end_sec < start_sec:
            end_sec = start_sec

        end_sec = max(end_sec, start_sec + min_duration)

        pos = np.where(
            (time_axis_sec >= start_sec) & (time_axis_sec <= end_sec)
        )[0]

        if pos.size == 0:
            center_sec = 0.5 * (start_sec + end_sec)
            nearest_idx = int(np.argmin(np.abs(time_axis_sec - center_sec)))
            spans[k, 0] = nearest_idx
            spans[k, 1] = nearest_idx
            valid[k] = 1.0
        else:
            spans[k, 0] = int(pos[0])
            spans[k, 1] = int(pos[-1])
            valid[k] = 1.0

    return spans.astype(np.int32), valid.astype(np.float32)

def normalize_gtscore(gtscore: np.ndarray) -> np.ndarray:
    gtscore = gtscore.astype(np.float32)
    gtscore = gtscore - gtscore.min()
    max_value = gtscore.max()
    if max_value > 0:
        gtscore = gtscore / max_value
    return gtscore