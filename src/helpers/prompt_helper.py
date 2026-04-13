from pathlib import Path
from typing import List


def load_prompt_vocabulary(path: Path) -> List[str]:
    prompts: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            prompts.append(text)

    if not prompts:
        raise ValueError(f'Empty prompt vocabulary: {path}')

    return prompts


def get_num_classes(path: Path) -> int:
    return len(load_prompt_vocabulary(path))