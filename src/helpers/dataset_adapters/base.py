from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class BasePseudoLabelAdapter(object):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def get_prompt_dataset_name(self) -> str:
        return self.dataset_name

    def resolve_items(self,
                      video_dir: str,
                      h5_path: str = None
                      ) -> List[Dict]:
        raise NotImplementedError

    @staticmethod
    def build_raw_video_index(video_dir: str) -> Dict[str, Path]:
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f'Video directory not found: {video_dir}')

        video_paths = sorted(video_dir.glob('*.mp4'))
        if not video_paths:
            raise ValueError(f'No .mp4 files found in: {video_dir}')

        index: Dict[str, Path] = {}
        for path in video_paths:
            key = path.stem
            if key in index:
                raise KeyError(f'Duplicate raw video stem: {key}')
            index[key] = path
        return index

    @staticmethod
    def pack_meta(item: Dict,
                  sampled_frame_count: int,
                  sampled_pick_count: int
                  ) -> Dict:
        return {
            'raw_video_name': item['raw_video_name'],
            'raw_video_file': str(item['video_path']),
            'review_status': item.get('review_status', 'SAFE'),
            'review_notes': item.get('review_notes'),
            'num_sampled_frames': int(sampled_frame_count),
            'sampled_picks_shape': int(sampled_pick_count),
        }