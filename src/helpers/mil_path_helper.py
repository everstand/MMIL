from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROMPTS_DIR = PROJECT_ROOT / 'prompts'
PSEUDO_LABELS_DIR = PROJECT_ROOT / 'pseudo_labels'
META_DIR = PROJECT_ROOT / 'meta'
FEATURES_DIR = PROJECT_ROOT / 'features'
CAPTIONS_DIR = PROJECT_ROOT / 'captions'
CAPTIONS_RAW_DIR = PROJECT_ROOT / 'captions_raw'


def normalize_dataset_name(dataset_name: str) -> str:
    name = dataset_name.strip().lower()
    name = re.sub(r'[^a-z0-9_]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name:
        raise ValueError(f'Invalid dataset name: {dataset_name}')
    return name


def get_prompt_path(dataset_name: str) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    return PROMPTS_DIR / f'{dataset_name}_prompt_vocabulary.txt'


def get_dataset_pseudo_dir(dataset_name: str) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    return PSEUDO_LABELS_DIR / dataset_name


def get_frame_text_scores_path(dataset_name: str) -> Path:
    return get_dataset_pseudo_dir(dataset_name) / 'frame_text_scores.npy'


def get_soft_labels_path(dataset_name: str) -> Path:
    return get_dataset_pseudo_dir(dataset_name) / 'soft_labels.npy'


def get_hard_labels_path(dataset_name: str) -> Path:
    return get_dataset_pseudo_dir(dataset_name) / 'hard_labels.npy'


def get_meta_path(dataset_name: str) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    return get_dataset_pseudo_dir(dataset_name) / 'meta.yaml'


def get_canonical_keys_path(dataset_name: str) -> Path:
    dataset_name = normalize_dataset_name(dataset_name)
    return META_DIR / f'{dataset_name}_canonical_keys.yaml'


def ensure_dataset_layout(dataset_name: str) -> None:
    dataset_dir = get_dataset_pseudo_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CAPTIONS_RAW_DIR.mkdir(parents=True, exist_ok=True)


def resolve_new_mainline_dataset_name(dataset_name: str) -> str:
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == 'summe':
        return 'summe24'
    return dataset_name


def get_openclip_feature_store_path(dataset_name: str) -> Path:
    dataset_name = resolve_new_mainline_dataset_name(dataset_name)
    return FEATURES_DIR / f'openclip_{dataset_name}.h5'


def get_dense_caption_json_path(dataset_name: str) -> Path:
    dataset_name = resolve_new_mainline_dataset_name(dataset_name)
    return CAPTIONS_DIR / f'{dataset_name}_dense_captions.json'

def get_structured_caption_json_path(dataset_name: str) -> Path:
    dataset_name = resolve_new_mainline_dataset_name(dataset_name)
    return CAPTIONS_RAW_DIR / f'{dataset_name}_dense_captions_structured.json'



def get_text_feature_store_path(dataset_name: str) -> Path:
    dataset_name = resolve_new_mainline_dataset_name(dataset_name)
    return FEATURES_DIR / f'text_{dataset_name}.h5'