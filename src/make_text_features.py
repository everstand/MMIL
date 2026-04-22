import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch

from helpers.mil_path_helper import (
    ensure_dataset_layout,
    get_dense_caption_json_path,
    get_openclip_feature_store_path,
    get_text_feature_store_path,
)
from helpers.openclip_helper import build_openclip_model, encode_texts


EXPECTED_FEATURE_DIM = 768


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=("summe", "tvsum"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--openclip-model",
        type=str,
        default="ViT-L-14",
    )
    parser.add_argument(
        "--openclip-pretrained",
        type=str,
        default="dfn2b",
    )

    return parser


def load_simple_caption_json(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Caption json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Caption json must be dict[h5_key -> List[str]], got {type(obj)}")

    normalized: Dict[str, List[str]] = {}
    for h5_key, captions in obj.items():
        if not isinstance(h5_key, str) or not h5_key.strip():
            raise ValueError(f"Invalid caption key: {h5_key}")

        if not isinstance(captions, list) or len(captions) == 0:
            raise ValueError(f"Captions for {h5_key} must be a non-empty list")

        cleaned: List[str] = []
        for idx, text in enumerate(captions):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Caption {idx} for {h5_key} is invalid")
            cleaned.append(text.strip())

        normalized[h5_key] = cleaned

    return normalized


def validate_key_alignment(
    caption_dict: Dict[str, List[str]],
    visual_feature_store_path: Path,
) -> List[str]:
    if not visual_feature_store_path.exists():
        raise FileNotFoundError(f"Visual feature store not found: {visual_feature_store_path}")

    with h5py.File(visual_feature_store_path, "r") as f:
        visual_keys = sorted(f.keys(), key=sort_video_key)

    caption_keys = sorted(caption_dict.keys(), key=sort_video_key)

    if visual_keys != caption_keys:
        visual_only = sorted(set(visual_keys) - set(caption_keys), key=sort_video_key)
        caption_only = sorted(set(caption_keys) - set(visual_keys), key=sort_video_key)
        raise ValueError(
            "Caption keys and visual feature keys are not aligned. "
            f"visual_only={visual_only[:10]}, caption_only={caption_only[:10]}"
        )

    return visual_keys


def sort_video_key(key: str):
    if key.startswith("video_"):
        suffix = key[len("video_"):]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, key)


@torch.no_grad()
def encode_caption_list(
    model,
    tokenizer,
    captions: List[str],
    device: str,
) -> np.ndarray:
    text_features = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=captions,
        device=device,
    )

    text_features = text_features.detach().cpu().numpy().astype(np.float32)

    if text_features.ndim != 2:
        raise ValueError(f"Expected text_features to be 2D [N, D], got {text_features.shape}")

    if text_features.shape[1] != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Unexpected text feature dim: got {text_features.shape[1]}, "
            f"expected {EXPECTED_FEATURE_DIM}"
        )

    return text_features


def main() -> None:
    args = get_parser().parse_args()

    ensure_dataset_layout(args.dataset)

    caption_json_path = get_dense_caption_json_path(args.dataset)
    visual_feature_store_path = get_openclip_feature_store_path(args.dataset)
    text_feature_store_path = get_text_feature_store_path(args.dataset)

    caption_dict = load_simple_caption_json(caption_json_path)
    ordered_keys = validate_key_alignment(
        caption_dict=caption_dict,
        visual_feature_store_path=visual_feature_store_path,
    )

    model, _, tokenizer = build_openclip_model(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained,
        device=args.device,
    )

    text_feature_store_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(text_feature_store_path, "w") as out_h5:
        out_h5.attrs["dataset"] = args.dataset
        out_h5.attrs["openclip_model"] = args.openclip_model
        out_h5.attrs["openclip_pretrained"] = args.openclip_pretrained
        out_h5.attrs["text_feature_dim"] = EXPECTED_FEATURE_DIM
        out_h5.attrs["text_target_rule"] = "mean_all_captions"
        out_h5.attrs["caption_source"] = str(caption_json_path)

        for idx, h5_key in enumerate(ordered_keys, start=1):
            captions = caption_dict[h5_key]
            all_text_features = encode_caption_list(
                model=model,
                tokenizer=tokenizer,
                captions=captions,
                device=args.device,
            )

            text_target = all_text_features.mean(axis=0).astype(np.float32)

            if text_target.ndim != 1 or text_target.shape[0] != EXPECTED_FEATURE_DIM:
                raise ValueError(
                    f"Invalid text_target shape for {h5_key}: {text_target.shape}"
                )

            group = out_h5.create_group(h5_key)
            group.create_dataset(
                "all_text_features",
                data=all_text_features,
                compression="gzip",
            )
            group.create_dataset(
                "text_target",
                data=text_target,
                compression="gzip",
            )
            group.attrs["num_captions"] = int(all_text_features.shape[0])

            print(
                f"[{idx}/{len(ordered_keys)}] Encoded {h5_key} | "
                f"num_captions={all_text_features.shape[0]} | "
                f"feature_dim={all_text_features.shape[1]}",
                flush=True,
            )

    print(f"[Done] text feature store written to: {text_feature_store_path}", flush=True)


if __name__ == "__main__":
    main()