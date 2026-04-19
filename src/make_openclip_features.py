import argparse
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from PIL import Image

from helpers.dataset_registry import get_pseudo_label_adapter
from helpers.mil_path_helper import (
    ensure_dataset_layout,
    get_openclip_feature_store_path,
)
from helpers.openclip_helper import (
    build_openclip_model,
    encode_images,
)
from helpers.video_text_align_helper import load_rgb_frames_by_picks


EXPECTED_FEATURE_DIM = 768


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=('tvsum', 'summe'),
    )
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--h5-path', type=str, required=True)

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=('cuda', 'cpu'),
    )
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--openclip-model', type=str, default='ViT-L-14')
    parser.add_argument('--openclip-pretrained', type=str, required=True)

    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--skip-broken', action='store_true')
    parser.add_argument('--audit-report', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output-h5', type=str, default=None)

    return parser


def main() -> None:
    args = get_parser().parse_args()

    ensure_dataset_layout(args.dataset)

    adapter = get_pseudo_label_adapter(args.dataset)
    items = adapter.resolve_items(
        video_dir=args.video_dir,
        h5_path=args.h5_path,
    )

    if args.limit is not None:
        items = items[:args.limit]

    feature_store_path = (
        Path(args.output_h5)
        if args.output_h5 is not None
        else get_openclip_feature_store_path(args.dataset)
    )
    audit_report_path = (
        Path(args.audit_report)
        if args.audit_report is not None
        else feature_store_path.with_suffix('.audit.json')
    )

    audit_report_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    preprocess = None

    if not args.audit_only:
        feature_store_path.parent.mkdir(parents=True, exist_ok=True)

        model, preprocess, _ = build_openclip_model(
            model_name=args.openclip_model,
            pretrained=args.openclip_pretrained,
            device=args.device,
        )

    failures = []
    success_count = 0

    with h5py.File(args.h5_path, 'r') as source_h5:
        if args.audit_only:
            feature_dim_written = None

            for item in items:
                h5_key = item['h5_key']

                try:
                    if h5_key not in source_h5:
                        raise KeyError(f'Missing h5 key in source h5: {h5_key}')

                    group = source_h5[h5_key]
                    if 'picks' not in group:
                        raise KeyError(f'Missing "picks" in source h5 group: {h5_key}')

                    picks = group['picks'][...].astype(np.int32)

                    loaded_picks, frames, audit = load_rgb_frames_by_picks(
                        video_path=item['video_path'],
                        picks=picks,
                    )

                    if len(frames) != len(loaded_picks):
                        raise ValueError(
                            f'Audit mismatch for {h5_key}: '
                            f'frames={len(frames)} vs picks={len(loaded_picks)}'
                        )

                    success_count += 1

                except Exception as exc:
                    failure_record = {
                        'h5_key': h5_key,
                        'raw_video_name': item['raw_video_name'],
                        'video_path': str(item['video_path']),
                        'review_status': item['review_status'],
                        'review_notes': item['review_notes'],
                        'error_type': type(exc).__name__,
                        'error_message': str(exc),
                    }
                    failures.append(failure_record)

        else:
            with h5py.File(feature_store_path, 'w') as target_h5:
                target_h5.attrs['dataset'] = args.dataset
                target_h5.attrs['openclip_model'] = args.openclip_model
                target_h5.attrs['openclip_pretrained'] = args.openclip_pretrained
                target_h5.attrs['time_axis_contract'] = 'strict_h5_picks_alignment'

                feature_dim_written = None

                for item in items:
                    h5_key = item['h5_key']

                    try:
                        if h5_key not in source_h5:
                            raise KeyError(f'Missing h5 key in source h5: {h5_key}')

                        group = source_h5[h5_key]
                        if 'picks' not in group:
                            raise KeyError(f'Missing "picks" in source h5 group: {h5_key}')

                        picks = group['picks'][...].astype(np.int32)

                        loaded_picks, frames, audit = load_rgb_frames_by_picks(
                            video_path=item['video_path'],
                            picks=picks,
                        )

                        features = encode_frames_as_openclip_features(
                            model=model,
                            preprocess=preprocess,
                            frames=frames,
                            device=args.device,
                            batch_size=args.batch_size,
                        )

                        if features.ndim != 2:
                            raise ValueError(
                                f'Expected OpenCLIP features to be 2D [T, D], got shape {features.shape} '
                                f'for {h5_key}'
                            )

                        if features.shape[0] != loaded_picks.shape[0]:
                            raise ValueError(
                                f'Feature length mismatch for {h5_key}: '
                                f'features={features.shape[0]} vs picks={loaded_picks.shape[0]}'
                            )

                        feature_dim = int(features.shape[1])

                        if feature_dim != EXPECTED_FEATURE_DIM:
                            raise ValueError(
                                f'Unexpected OpenCLIP feature dim for {h5_key}: '
                                f'got {feature_dim}, expected {EXPECTED_FEATURE_DIM}'
                            )

                        if feature_dim_written is None:
                            feature_dim_written = feature_dim
                            target_h5.attrs['feature_dim'] = feature_dim_written
                        elif feature_dim != feature_dim_written:
                            raise ValueError(
                                f'Inconsistent OpenCLIP feature dim for {h5_key}: '
                                f'got {feature_dim}, expected {feature_dim_written}'
                            )

                        out_group = target_h5.create_group(h5_key)
                        out_group.create_dataset(
                            'features',
                            data=features.astype(np.float32),
                            compression='gzip',
                        )

                        out_group.attrs['raw_video_name'] = item['raw_video_name']
                        out_group.attrs['review_status'] = item['review_status']
                        out_group.attrs['review_notes'] = '' if item['review_notes'] is None else str(item['review_notes'])
                        out_group.attrs['requested_picks'] = int(audit['requested_picks'])
                        out_group.attrs['decoded_frames'] = int(audit['decoded_frames'])
                        out_group.attrs['decode_failures'] = int(audit['decode_failures'])
                        out_group.attrs['first_pick'] = int(audit['first_pick'])
                        out_group.attrs['last_pick'] = int(audit['last_pick'])
                        out_group.attrs['last_scanned_frame'] = int(audit['last_scanned_frame'])
                        out_group.attrs['reported_total_frames'] = int(audit['reported_total_frames'])
                        out_group.attrs['reported_width'] = int(audit['reported_width'])
                        out_group.attrs['reported_height'] = int(audit['reported_height'])
                        out_group.attrs['reported_fps'] = int(audit['reported_fps'])

                        success_count += 1

                    except Exception as exc:
                        failure_record = {
                            'h5_key': h5_key,
                            'raw_video_name': item['raw_video_name'],
                            'video_path': str(item['video_path']),
                            'review_status': item['review_status'],
                            'review_notes': item['review_notes'],
                            'error_type': type(exc).__name__,
                            'error_message': str(exc),
                        }
                        failures.append(failure_record)

                        if not args.skip_broken:
                            raise

    report = {
        'dataset': args.dataset,
        'openclip_model': args.openclip_model,
        'openclip_pretrained': args.openclip_pretrained,
        'audit_only': bool(args.audit_only),
        'skip_broken': bool(args.skip_broken),
        'num_items': len(items),
        'num_success': int(success_count),
        'num_failures': int(len(failures)),
        'failures': failures,
    }

    with open(audit_report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if failures:
        print(f'[Audit] failures={len(failures)} report={audit_report_path}')
    else:
        print(f'[Audit] all passed report={audit_report_path}')

    if args.audit_only:
        return

    if failures and args.skip_broken:
        print(
            '[Warning] Partial feature store written because --skip-broken was enabled. '
            'Do NOT use it for formal training.'
        )

    print(f'[Done] OpenCLIP feature store written to: {feature_store_path}')


def encode_frames_as_openclip_features(
    model,
    preprocess,
    frames: List[np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    if len(frames) == 0:
        raise ValueError('encode_frames_as_openclip_features received empty frame list.')

    with torch.no_grad():
        feature_batches = []

        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            batch_tensor = torch.stack(
                [preprocess(Image.fromarray(frame)) for frame in batch_frames],
                dim=0,
            ).to(device)

            batch_features = encode_images(model, batch_tensor)
            feature_batches.append(batch_features)

        features = torch.cat(feature_batches, dim=0)

    return features.detach().cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    main()