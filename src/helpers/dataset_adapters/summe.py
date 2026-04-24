from typing import Dict, List

import h5py

from helpers.dataset_adapters.base import BasePseudoLabelAdapter
from helpers.key_helper import decode_h5_string


class SumMePseudoLabelAdapter(BasePseudoLabelAdapter):
    def __init__(self):
        super().__init__(dataset_name='summe')

    def resolve_items(self,
                      video_dir: str,
                      h5_path: str = None
                      ) -> List[Dict]:
        if h5_path is None:
            raise ValueError('SumMe adapter requires --h5-path')

        raw_video_index = self.build_raw_video_index(video_dir)
        items: List[Dict] = []

        with h5py.File(h5_path, 'r') as h5_file:
            h5_keys = sorted(h5_file.keys(), key=self.parse_h5_group_index)

            for h5_key in h5_keys:
                group = h5_file[h5_key]
                if 'video_name' not in group:
                    raise KeyError(f'Missing "video_name" in SumMe h5 group: {h5_key}')

                raw_video_name = decode_h5_string(group['video_name'][()])
                if raw_video_name not in raw_video_index:
                    raise KeyError(
                        f'Missing raw SumMe video for h5 key {h5_key}: {raw_video_name}'
                    )

                items.append({
                    'h5_key': h5_key,
                    'raw_video_name': raw_video_name,
                    'video_path': raw_video_index[raw_video_name],
                    'review_status': 'SAFE',
                    'review_notes': None,
                })

        if len(items) != 25:
            raise ValueError(
                f'Standard SumMe-25 adapter expected 25 items, got {len(items)}. '
                'Check raw video availability and h5/video_name alignment.'
            )

        return items

    @staticmethod
    def parse_h5_group_index(h5_key: str) -> int:
        prefix = 'video_'
        if not h5_key.startswith(prefix):
            raise ValueError(f'Unexpected SumMe h5 key: {h5_key}')
        suffix = h5_key[len(prefix):]
        if not suffix.isdigit():
            raise ValueError(f'Unexpected SumMe h5 key index: {h5_key}')
        return int(suffix)
