import argparse
from typing import Dict

import numpy as np
import torch
import yaml
from PIL import Image

from helpers.dataset_registry import get_pseudo_label_adapter
from helpers.mil_path_helper import (
    ensure_dataset_layout,
    get_frame_text_scores_path,
    get_meta_path,
    get_prompt_path,
    get_soft_labels_path,
)
from helpers.openclip_helper import (
    build_openclip_model,
    compute_similarity,
    encode_images,
    encode_prompts,
)
from helpers.prompt_helper import load_prompt_vocabulary
from helpers.pseudo_label_helper import (
    aggregate_soft_labels,
    package_pseudo_label_record,
    temporal_window_smooth,
)
from helpers.video_text_align_helper import load_sampled_rgb_frames_with_audit


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=('tvsum', 'summe'))
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--h5-path', type=str, default=None)

    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--sample-rate', type=int, default=15)
    parser.add_argument('--temporal-radius', type=int, default=1)
    parser.add_argument('--soft-top-ratio', type=float, default=0.15)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--openclip-model', type=str, default='ViT-L-14')
    parser.add_argument('--openclip-pretrained', type=str, required=True)

    return parser


def main() -> None:
    args = get_parser().parse_args()

    adapter = get_pseudo_label_adapter(args.dataset)
    dataset_name = adapter.dataset_name

    ensure_dataset_layout(dataset_name)

    prompt_path = get_prompt_path(adapter.get_prompt_dataset_name())
    prompts = load_prompt_vocabulary(prompt_path)

    model, preprocess, tokenizer = build_openclip_model(
        model_name=args.openclip_model,
        pretrained=args.openclip_pretrained,
        device=args.device,
    )
    text_features = encode_prompts(model, tokenizer, prompts, args.device)
    logit_scale = getattr(model, 'logit_scale', None)

    items = adapter.resolve_items(
        video_dir=args.video_dir,
        h5_path=args.h5_path,
    )

    frame_text_scores_dict: Dict[str, np.ndarray] = {}
    soft_labels_dict: Dict[str, np.ndarray] = {}
    video_meta: Dict[str, Dict] = {}

    for item in items:
        _, sampled_picks, sampled_frames, decode_audit = load_sampled_rgb_frames_with_audit(
            video_path=item['video_path'],
            sample_rate=args.sample_rate,
        )

        frame_text_scores = compute_frame_text_scores(
            model=model,
            preprocess=preprocess,
            text_features=text_features,
            logit_scale=logit_scale,
            frames=sampled_frames,
            device=args.device,
            batch_size=args.batch_size,
            temporal_radius=args.temporal_radius,
        )
        soft_labels = aggregate_soft_labels(
            frame_text_scores=frame_text_scores,
            top_ratio=args.soft_top_ratio,
        )

        record = package_pseudo_label_record(
            frame_text_scores=frame_text_scores,
            soft_labels=soft_labels,
        )

        h5_key = item['h5_key']
        frame_text_scores_dict[h5_key] = record['frame_text_scores']
        soft_labels_dict[h5_key] = record['soft_labels']

        packed_meta = adapter.pack_meta(
            item=item,
            sampled_frame_count=len(sampled_frames),
            sampled_pick_count=sampled_picks.shape[0],
        )
        packed_meta.update({
            'decode_failures': int(decode_audit['decode_failures']),
            'expected_sampled_frames': int(decode_audit['expected_sampled_frames']),
            'valid_sampled_frames': int(decode_audit['valid_sampled_frames']),
        })
        video_meta[h5_key] = packed_meta

        flush_outputs(
            dataset_name=dataset_name,
            frame_text_scores_dict=frame_text_scores_dict,
            soft_labels_dict=soft_labels_dict,
            global_meta=build_global_meta(
                dataset_name=dataset_name,
                prompt_path=str(prompt_path),
                num_classes=len(prompts),
                num_videos=len(soft_labels_dict),
                sample_rate=args.sample_rate,
                temporal_radius=args.temporal_radius,
                soft_top_ratio=args.soft_top_ratio,
                openclip_model=args.openclip_model,
                openclip_pretrained=args.openclip_pretrained,
                videos=video_meta,
            ),
        )


def flush_outputs(dataset_name: str,
                  frame_text_scores_dict: Dict[str, np.ndarray],
                  soft_labels_dict: Dict[str, np.ndarray],
                  global_meta: Dict
                  ) -> None:
    np.save(
        get_frame_text_scores_path(dataset_name),
        frame_text_scores_dict,
        allow_pickle=True,
    )
    np.save(
        get_soft_labels_path(dataset_name),
        soft_labels_dict,
        allow_pickle=True,
    )
    with open(get_meta_path(dataset_name), 'w', encoding='utf-8') as f:
        yaml.safe_dump(global_meta, f, sort_keys=True, allow_unicode=True)


def build_global_meta(dataset_name: str,
                      prompt_path: str,
                      num_classes: int,
                      num_videos: int,
                      sample_rate: int,
                      temporal_radius: int,
                      soft_top_ratio: float,
                      openclip_model: str,
                      openclip_pretrained: str,
                      videos: Dict[str, Dict]
                      ) -> Dict:
    return {
        'dataset': dataset_name,
        'prompt_path': prompt_path,
        'num_classes': num_classes,
        'num_videos': num_videos,
        'sample_rate': sample_rate,
        'temporal_radius': temporal_radius,
        'soft_top_ratio': soft_top_ratio,
        'openclip_model': openclip_model,
        'openclip_pretrained': openclip_pretrained,
        'label_key_contract': 'h5_key',
        'time_axis_contract': 'raw_video_sampling_only_not_aligned_to_training_h5',
        'hard_label_rule': '[缺失/待确认]',
        'videos': videos,
    }


def compute_frame_text_scores(model,
                              preprocess,
                              text_features: torch.Tensor,
                              logit_scale,
                              frames,
                              device: str,
                              batch_size: int,
                              temporal_radius: int
                              ) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError('Empty sampled frame list')

    with torch.no_grad():
        image_feature_batches = []

        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            batch_tensor = torch.stack(
                [preprocess(Image.fromarray(frame)) for frame in batch_frames],
                dim=0,
            ).to(device)

            batch_features = encode_images(model, batch_tensor)
            image_feature_batches.append(batch_features)

        image_features = torch.cat(image_feature_batches, dim=0)
        image_features = temporal_window_smooth(
            frame_features=image_features,
            radius=temporal_radius,
        )

        scores = compute_similarity(
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )

    return scores.detach().cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    main()