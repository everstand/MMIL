from typing import Dict, List

from helpers.dataset_adapters.base import BasePseudoLabelAdapter
from helpers.key_helper import canonicalize_video_name
from helpers.tvsum_metadata import TVSUM_STATIC_MAP, TVSUM_REVIEW_NOTES


class TVSumPseudoLabelAdapter(BasePseudoLabelAdapter):
    def __init__(self):
        super().__init__(dataset_name='tvsum')

    def resolve_items(self,
                      video_dir: str,
                      h5_path: str = None
                      ) -> List[Dict]:
        raw_video_index = self.build_raw_video_index(video_dir)
        items: List[Dict] = []

        for h5_key, raw_video_name in sorted(
            TVSUM_STATIC_MAP.items(),
            key=lambda x: self.parse_h5_group_index(x[0])
        ):
            canonical_name = canonicalize_video_name(raw_video_name)
            if canonical_name not in raw_video_index:
                raise KeyError(
                    f'Missing raw video for mapped TVSum key: {h5_key} -> {raw_video_name}'
                )

            review_info = TVSUM_REVIEW_NOTES.get(h5_key)
            items.append({
                'h5_key': h5_key,
                'raw_video_name': raw_video_name,
                'video_path': raw_video_index[canonical_name],
                'review_status': 'REVIEW' if review_info is not None else 'SAFE',
                'review_notes': review_info,
            })

        return items

    @staticmethod
    def parse_h5_group_index(h5_key: str) -> int:
        prefix = 'video_'
        if not h5_key.startswith(prefix):
            raise ValueError(f'Unexpected TVSum h5 key: {h5_key}')
        suffix = h5_key[len(prefix):]
        if not suffix.isdigit():
            raise ValueError(f'Unexpected TVSum h5 key index: {h5_key}')
        return int(suffix)