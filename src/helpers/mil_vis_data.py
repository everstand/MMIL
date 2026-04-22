import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from anchor_free.dsnet_af_mil import DSNetAFMIL
from anchor_free.dsnet_af_mil_cond import DSNetAFMILCond
from helpers import vsumm_helper
from helpers.mil_data_helper import VideoDatasetMIL
from helpers.mil_data_helper_cond import VideoDatasetMILCond


def build_key(h5_path: str, h5_key: str) -> str:
    return str(Path(h5_path).resolve() / h5_key)


def normalize_split_key(key: str, split_root: Path) -> str:
    key_path = Path(key)
    h5_rel_path = key_path.parent
    h5_group_name = key_path.name
    h5_abs_path = (split_root / h5_rel_path).resolve()

    if not h5_abs_path.exists():
        raise FileNotFoundError(
            f'Normalized HDF5 path does not exist: {h5_abs_path} '
            f'(from split key "{key}")'
        )

    return str(h5_abs_path / h5_group_name)


def load_all_splits(split_yaml: str) -> List[Dict]:
    path = Path(split_yaml)
    if not path.exists():
        raise FileNotFoundError(f"Missing split yaml: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, list):
        raise ValueError(f"Split yaml must contain a list, got {type(obj)}")

    split_root = path.parent.resolve()
    normalized = []

    for idx, split in enumerate(obj):
        if not isinstance(split, dict):
            raise ValueError(f"Invalid split entry at index {idx}")
        if "train_keys" not in split or "test_keys" not in split:
            raise ValueError(f"Split entry {idx} missing train_keys/test_keys")

        normalized.append({
            "train_keys": [normalize_split_key(x, split_root) for x in split["train_keys"]],
            "test_keys": [normalize_split_key(x, split_root) for x in split["test_keys"]],
        })

    return normalized


def infer_dataset_name_from_key(key: str) -> str:
    key_lower = str(key).lower()
    if "tvsum" in key_lower:
        return "tvsum"
    if "summe" in key_lower:
        return "summe"
    raise ValueError(f"Cannot infer dataset from key: {key}")


def select_keys_from_split(split_yaml: str,
                           split_index: int,
                           subset: str,
                           dataset: str,
                           max_videos: int = None) -> List[str]:
    splits = load_all_splits(split_yaml)

    if split_index < 0 or split_index >= len(splits):
        raise IndexError(
            f"split-index out of range: {split_index}, num_splits={len(splits)}"
        )

    split = splits[split_index]
    key_field = "train_keys" if subset == "train" else "test_keys"
    keys = split[key_field]

    for key in keys:
        dataset_name = infer_dataset_name_from_key(key)
        if dataset_name != dataset:
            raise ValueError(
                f"Dataset mismatch: expected {dataset}, got {dataset_name} from key {key}"
            )

    if max_videos is not None:
        keys = keys[:max_videos]
    return keys


def load_baseline_scores(h5_path: str,
                         h5_key: str,
                         ckpt_path: str,
                         device: str,
                         base_model: str,
                         num_feature: int,
                         num_hidden: int,
                         num_head: int) -> np.ndarray:
    key = build_key(h5_path, h5_key)
    ds = VideoDatasetMIL([key])
    item = ds[0]

    seq = item[1]
    soft_label = item[2]
    num_classes = int(np.asarray(soft_label).shape[0])

    model = DSNetAFMIL(
        base_model=base_model,
        num_feature=num_feature,
        num_hidden=num_hidden,
        num_head=num_head,
        num_classes=num_classes,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, attn_weights, _ = model(seq_tensor)

    scores = attn_weights.detach().cpu().numpy().astype(np.float32)
    return scores


def load_ours_scores(h5_path: str,
                     h5_key: str,
                     ckpt_path: str,
                     device: str,
                     base_model: str,
                     num_feature: int,
                     num_hidden: int,
                     num_head: int,
                     text_cond_num: int) -> np.ndarray:
    key = build_key(h5_path, h5_key)
    ds = VideoDatasetMILCond([key], text_cond_num=text_cond_num, random_text_sampling=False)
    item = ds[0]

    seq = item[1]
    soft_label = item[2]
    text_cond = item[3]
    num_classes = int(np.asarray(soft_label).shape[0])

    model = DSNetAFMILCond(
        base_model=base_model,
        num_feature=num_feature,
        num_hidden=num_hidden,
        num_head=num_head,
        num_classes=num_classes,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, attn_weights, _, _ = model(seq_tensor, text_cond_tensor)

    scores = attn_weights.detach().cpu().numpy().astype(np.float32)
    return scores


def load_baseline_summary(h5_path: str,
                          h5_key: str,
                          ckpt_path: str,
                          device: str,
                          base_model: str,
                          num_feature: int,
                          num_hidden: int,
                          num_head: int):
    key = build_key(h5_path, h5_key)
    ds = VideoDatasetMIL([key])
    item = ds[0]

    seq = item[1]
    soft_label = item[2]
    user_summary = item[4]
    cps = item[5]
    n_frames = int(np.asarray(item[6]).item())
    nfps = item[7]
    picks = item[8]
    num_classes = int(np.asarray(soft_label).shape[0])

    model = DSNetAFMIL(
        base_model=base_model,
        num_feature=num_feature,
        num_hidden=num_hidden,
        num_head=num_head,
        num_classes=num_classes,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, attn_weights, _ = model(seq_tensor)

    scores = attn_weights.detach().cpu().numpy().astype(np.float32)
    pred_summ = vsumm_helper.get_keyshot_summ(scores, cps, n_frames, nfps, picks)
    return pred_summ, user_summary, n_frames


def load_ours_summary(h5_path: str,
                      h5_key: str,
                      ckpt_path: str,
                      device: str,
                      base_model: str,
                      num_feature: int,
                      num_hidden: int,
                      num_head: int,
                      text_cond_num: int):
    key = build_key(h5_path, h5_key)
    ds = VideoDatasetMILCond([key], text_cond_num=text_cond_num, random_text_sampling=False)
    item = ds[0]

    seq = item[1]
    soft_label = item[2]
    text_cond = item[3]
    user_summary = item[6]
    cps = item[7]
    n_frames = int(np.asarray(item[8]).item())
    nfps = item[9]
    picks = item[10]
    num_classes = int(np.asarray(soft_label).shape[0])

    model = DSNetAFMILCond(
        base_model=base_model,
        num_feature=num_feature,
        num_hidden=num_hidden,
        num_head=num_head,
        num_classes=num_classes,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    text_cond_tensor = torch.tensor(text_cond, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, attn_weights, _, _ = model(seq_tensor, text_cond_tensor)

    scores = attn_weights.detach().cpu().numpy().astype(np.float32)
    pred_summ = vsumm_helper.get_keyshot_summ(scores, cps, n_frames, nfps, picks)
    return pred_summ, user_summary, n_frames


def choose_gt_summary(user_summary: np.ndarray) -> np.ndarray:
    user_summary = np.asarray(user_summary, dtype=np.float32)
    if user_summary.ndim == 1:
        return user_summary
    return user_summary[0]