from pathlib import Path
from typing import Union


def canonicalize_video_name(name: Union[str, Path]) -> str:
    raw = Path(str(name)).stem.strip()
    if raw.endswith('_fixed'):
        raw = raw[:-6]
    return raw


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if hasattr(value, 'decode'):
        return value.decode('utf-8')
    return str(value)