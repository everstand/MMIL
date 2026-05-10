# -*- coding: utf-8 -*-
"""Build an LLM preference teacher from shot-indexed captions.

The LLM returns perspective_scores only. Final summary masks are produced by the
builder with shot lengths and the fixed summary budget.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

from helpers.mil_data_helper_cond import extract_caption_items, extract_video_fps, parse_mmss
from helpers.preference_teacher_helper import (
    CANONICAL_PERSPECTIVE_IDS,
    build_preference_teacher_record,
)

logger = logging.getLogger(__name__)


ALLOWED_ROOT_KEYS = {'perspectives'}
ALLOWED_PERSPECTIVE_KEYS = {'perspective_id', 'scores', 'selected_shot_indices_hint', 'selected_segment_indices_hint'}
ALLOWED_SCORE_KEYS = {'segment_index', 'shot_index', 'score', 'brief_reason'}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='summe', choices=('summe',))
    parser.add_argument(
        '--structured-caption-path',
        type=str,
        default='captions_raw/summe_dense_captions_structured_v3.json',
    )
    parser.add_argument(
        '--h5-path',
        type=str,
        default='datasets/eccv16_dataset_summe_google_pool5.h5',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='pseudo_labels/summe/llm_preference_teacher_v1.npy',
    )
    parser.add_argument('--failures-output', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='gemini-3.1-pro-preview')
    parser.add_argument('--base-url', type=str, default='https://api.xheai.cc/v1/')
    parser.add_argument('--api-key-env', type=str, default='OPENAI_API_KEY')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--retry-wait-seconds', type=float, default=2.0)
    parser.add_argument('--disable-response-format-json', action='store_true')
    parser.add_argument('--num-perspectives', type=int, default=7)
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--max-pairs-per-video', type=int, default=96)
    parser.add_argument('--pair-seed', type=int, default=19500)
    parser.add_argument('--positive-threshold', type=float, default=0.60)
    parser.add_argument('--negative-threshold', type=float, default=0.20)
    parser.add_argument('--max-captions-per-shot', type=int, default=4)
    parser.add_argument('--max-caption-chars', type=int, default=180)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--only-keys', type=str, nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser


def natural_key(name: str):
    parts = re.split(r'(\d+)', str(name))
    return tuple(int(p) if p.isdigit() else p for p in parts)


def load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected structured caption JSON dict, got {type(obj)}')
    return obj


def save_records(path: Path, records: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), records, allow_pickle=True)


def save_failures(path: Path, failures: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(failures, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_existing_records(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    obj = np.load(str(path), allow_pickle=True)
    try:
        obj = obj.item()
    except Exception as exc:
        raise ValueError(f'Invalid existing teacher file: {path}') from exc
    if not isinstance(obj, dict):
        raise ValueError(f'Existing teacher file must contain a dict, got {type(obj)}')
    return obj


def sanitize_caption(text: str, max_chars: int) -> str:
    text = ' '.join(str(text).strip().split())
    replacements = {
        'United States of America': 'country name text',
        'United States': 'country name text',
        'U.S.': 'country name text',
        'USA': 'country name text',
        'about to make contact with the runway': 'very close to the runway',
        'make contact with the runway': 'reach the runway',
        'touches down on the runway': 'reaches the runway',
        'touching down on the runway': 'reaching the runway',
        'point of touchdown': 'landing moment',
        'touchdown': 'landing moment',
        'puff of smoke': 'brief landing dust',
        'creating a small puff of smoke': 'showing a visible landing effect',
        'creating a small brief landing dust': 'showing a visible landing effect',
        'brief landing dust': 'visible landing effect',
        'small brief landing dust': 'visible landing effect',
        'smoke': 'visual effect',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + '...'


def caption_frame_range(item: Dict, fps: float, total_frames: int) -> Tuple[int, int]:
    start_sec = parse_mmss(item['start_mmss'])
    end_sec = parse_mmss(item['end_mmss'])
    if end_sec < start_sec:
        end_sec = start_sec
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    return start_frame, end_frame


def ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def build_shot_indexed_entries(h5_key: str,
                               structured_entry: Dict,
                               cps: np.ndarray,
                               nfps: np.ndarray,
                               n_frames: int,
                               max_captions_per_shot: int,
                               max_caption_chars: int) -> List[Dict]:
    captions = extract_caption_items(structured_entry)
    fps = extract_video_fps(structured_entry, h5_key)
    caption_ranges = []
    for item in captions:
        start_frame, end_frame = caption_frame_range(item, fps=fps, total_frames=int(n_frames))
        caption_ranges.append((start_frame, end_frame, item))

    shot_entries = []
    for shot_idx, (first, last) in enumerate(np.asarray(cps, dtype=np.int64).tolist()):
        overlaps = []
        for cap_start, cap_end, item in caption_ranges:
            if ranges_overlap(first, last, cap_start, cap_end):
                overlaps.append(sanitize_caption(item.get('caption', ''), max_caption_chars))
        if len(overlaps) > max_captions_per_shot:
            keep = overlaps[:max_captions_per_shot]
            keep.append(f'(+{len(overlaps) - max_captions_per_shot} more caption spans)')
            overlaps = keep
        duration_sec = float(np.asarray(nfps, dtype=np.float32).reshape(-1)[shot_idx]) / float(fps)
        shot_entries.append({
            'shot_index': int(shot_idx),
            'frame_start': int(first),
            'frame_end': int(last),
            'duration_sec': round(duration_sec, 3),
            'captions': overlaps if overlaps else ['No caption overlap.'],
        })
    return shot_entries


def build_prompt(h5_key: str,
                 shot_entries: Sequence[Dict],
                 perspective_ids: Sequence[str],
                 summary_budget: float,
                 n_frames: int) -> List[Dict[str, str]]:
    system = (
        'You produce weak segment-level preference scores for video summary research. '
        'Use only the shot-indexed captions and metadata in the request. '
        'Score every segment for every requested perspective. Return strict JSON only.'
    )
    schema = {
        'perspectives': [
            {
                'perspective_id': perspective_ids[0],
                'scores': [
                    {'segment_index': 0, 'score': 0.0, 'brief_reason': 'short reason'}
                ],
                'selected_segment_indices_hint': [0],
            }
        ]
    }
    shot_lines = []
    for entry in shot_entries:
        captions = ' | '.join(entry['captions'])
        shot_lines.append(
            f"Segment {entry['shot_index']}: frames {entry['frame_start']}-{entry['frame_end']}; "
            f"duration {entry['duration_sec']:.3f}s; captions: {captions}"
        )
    user = (
        f'Video key: {h5_key}\n'
        f'Number of segments: {len(shot_entries)}\n'
        f'Total frames: {int(n_frames)}\n'
        f'Summary budget reference: {float(summary_budget):.3f} of the video. '
        'Do not output final masks; only output scores.\n\n'
        'Perspective IDs, in exact order:\n'
        + '\n'.join(f'- {pid}' for pid in perspective_ids)
        + '\n\nScoring instructions:\n'
        '- For each perspective, score every segment from 0.0 to 1.0.\n'
        '- Higher score means the segment is more useful for a concise viewer-facing summary under that perspective.\n'
        '- Prefer story-progressing, visually distinctive, memorable, event-relevant, non-redundant segments.\n'
        '- Suppress repetitive background, idle setup, and near-duplicate shots unless they are necessary for coherence.\n'
        '- selected_segment_indices_hint is optional guidance only and may be approximate.\n'
        '- Return exactly one JSON object with no markdown and no extra fields.\n\n'
        f'Required JSON schema example:\n{json.dumps(schema, ensure_ascii=False)}\n\n'
        'Segment-indexed input, where each segment has a fixed segment_index:\n'
        + '\n'.join(shot_lines)
    )
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]


def make_client(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f'Missing API key environment variable: {args.api_key_env}')
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=args.base_url)


def request_llm(client, args, messages: List[Dict[str, str]]) -> str:
    kwargs = {
        'model': args.model_name,
        'messages': messages,
        'temperature': float(args.temperature),
    }
    if not args.disable_response_format_json:
        kwargs['response_format'] = {'type': 'json_object'}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Empty LLM response content.')
    return content.strip()


def is_sensitive_filter_error(exc: Exception) -> bool:
    return 'sensitive_words_detected' in repr(exc)


def make_metadata_only_entries(shot_entries: Sequence[Dict]) -> List[Dict]:
    safe_entries = []
    for entry in shot_entries:
        safe = dict(entry)
        safe['captions'] = ['visual segment metadata only']
        safe_entries.append(safe)
    return safe_entries


def parse_and_validate_response(content: str,
                                perspective_ids: Sequence[str],
                                num_shots: int) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    stripped = content.strip()
    if stripped.startswith('```') or stripped.endswith('```'):
        raise ValueError('Response used markdown fencing; strict JSON is required.')
    obj = json.loads(stripped)
    if not isinstance(obj, dict):
        raise ValueError(f'Response root must be object, got {type(obj)}')
    extra_root = set(obj.keys()) - ALLOWED_ROOT_KEYS
    missing_root = ALLOWED_ROOT_KEYS - set(obj.keys())
    if extra_root or missing_root:
        raise ValueError(f'Invalid root fields: extra={sorted(extra_root)}, missing={sorted(missing_root)}')

    perspectives = obj['perspectives']
    if not isinstance(perspectives, list):
        raise ValueError('perspectives must be a list.')
    if len(perspectives) != len(perspective_ids):
        raise ValueError(f'Perspective count mismatch: {len(perspectives)} vs {len(perspective_ids)}')

    expected_ids = list(perspective_ids)
    seen_ids = []
    scores = np.zeros((len(expected_ids), int(num_shots)), dtype=np.float32)
    selected_hints: Dict[str, List[int]] = {}

    for k, item in enumerate(perspectives):
        if not isinstance(item, dict):
            raise ValueError(f'Perspective entry {k} must be object.')
        extra = set(item.keys()) - ALLOWED_PERSPECTIVE_KEYS
        missing = {'perspective_id', 'scores'} - set(item.keys())
        if extra or missing:
            raise ValueError(
                f'Invalid fields for perspective {k}: extra={sorted(extra)}, missing={sorted(missing)}'
            )
        pid = item['perspective_id']
        if pid != expected_ids[k]:
            raise ValueError(f'Perspective ID/order mismatch at {k}: {pid} vs {expected_ids[k]}')
        if pid in seen_ids:
            raise ValueError(f'Duplicate perspective_id: {pid}')
        seen_ids.append(pid)

        score_items = item['scores']
        if not isinstance(score_items, list) or len(score_items) != int(num_shots):
            raise ValueError(f'scores for {pid} must contain exactly {num_shots} entries.')
        seen_shots = set()
        for score_item in score_items:
            if not isinstance(score_item, dict):
                raise ValueError(f'score entry for {pid} must be object.')
            extra_score = set(score_item.keys()) - ALLOWED_SCORE_KEYS
            has_index = 'segment_index' in score_item or 'shot_index' in score_item
            missing_score = {'score'} - set(score_item.keys())
            if extra_score or missing_score or not has_index:
                missing = sorted(missing_score | (set() if has_index else {'segment_index'}))
                raise ValueError(
                    f'Invalid score fields for {pid}: '
                    f'extra={sorted(extra_score)}, missing={missing}'
                )
            shot_index = int(score_item.get('segment_index', score_item.get('shot_index')))
            if shot_index < 0 or shot_index >= int(num_shots):
                raise ValueError(f'Invalid shot_index={shot_index} for {pid}')
            if shot_index in seen_shots:
                raise ValueError(f'Duplicate shot_index={shot_index} for {pid}')
            value = float(score_item['score'])
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f'Invalid score={value} for {pid} shot {shot_index}')
            scores[k, shot_index] = value
            seen_shots.add(shot_index)
        if seen_shots != set(range(int(num_shots))):
            raise ValueError(f'Missing shot scores for {pid}')

        hints = item.get('selected_segment_indices_hint', item.get('selected_shot_indices_hint', []))
        if hints is None:
            hints = []
        if not isinstance(hints, list):
            raise ValueError(f'selected_shot_indices_hint for {pid} must be a list.')
        clean_hints = []
        for shot_index in hints:
            idx = int(shot_index)
            if idx < 0 or idx >= int(num_shots):
                raise ValueError(f'Invalid selected_shot_indices_hint={idx} for {pid}')
            clean_hints.append(idx)
        selected_hints[pid] = sorted(set(clean_hints))

    return scores.astype(np.float32), selected_hints


def process_video(client, args, h5_key: str, h5_group, structured_entry: Dict, perspective_ids: Sequence[str]) -> Dict:
    cps = h5_group['change_points'][...].astype(np.int32)
    nfps = h5_group['n_frame_per_seg'][...].astype(np.int32)
    n_frames = int(np.asarray(h5_group['n_frames'][...]).item())
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Invalid change_points shape for {h5_key}: {cps.shape}')
    if nfps.shape[0] != cps.shape[0]:
        raise ValueError(f'nfps/cps mismatch for {h5_key}: {nfps.shape[0]} vs {cps.shape[0]}')

    shot_entries = build_shot_indexed_entries(
        h5_key=h5_key,
        structured_entry=structured_entry,
        cps=cps,
        nfps=nfps,
        n_frames=n_frames,
        max_captions_per_shot=args.max_captions_per_shot,
        max_caption_chars=args.max_caption_chars,
    )
    prompt_perspective_ids = [f'p{i}' for i in range(len(perspective_ids))]
    entry_variants = [
        ('caption_full', shot_entries),
        ('metadata_only', make_metadata_only_entries(shot_entries)),
    ]

    last_error = None
    for caption_safety_mode, entries_for_prompt in entry_variants:
        messages = build_prompt(
            h5_key=h5_key,
            shot_entries=entries_for_prompt,
            perspective_ids=prompt_perspective_ids,
            summary_budget=args.summary_budget,
            n_frames=n_frames,
        )
        for attempt in range(int(args.max_retries) + 1):
            try:
                content = request_llm(client, args, messages)
                perspective_scores, selected_hints = parse_and_validate_response(
                    content=content,
                    perspective_ids=prompt_perspective_ids,
                    num_shots=cps.shape[0],
                )
                meta = {
                    'dataset': args.dataset,
                    'h5_key': h5_key,
                    'num_shots': int(cps.shape[0]),
                    'n_frames': n_frames,
                    'summary_budget': float(args.summary_budget),
                    'capacity_frames': int(n_frames * float(args.summary_budget)),
                    'canonical_perspective_ids': list(perspective_ids),
                    'prompt_perspective_ids': list(prompt_perspective_ids),
                    'caption_safety_mode': caption_safety_mode,
                    'selected_shot_indices_hint': selected_hints,
                    'model_name': args.model_name,
                    'base_url': args.base_url,
                    'pair_seed': int(args.pair_seed),
                    'max_pairs_per_video': int(args.max_pairs_per_video),
                    'positive_threshold': float(args.positive_threshold),
                    'negative_threshold': float(args.negative_threshold),
                    'builder': 'make_llm_preference_teacher.py',
                    'llm_attempt': int(attempt + 1),
                }
                return build_preference_teacher_record(
                    perspective_scores=perspective_scores,
                    nfps=nfps,
                    n_frames=n_frames,
                    summary_budget=args.summary_budget,
                    max_pairs_per_video=args.max_pairs_per_video,
                    pair_seed=args.pair_seed,
                    positive_threshold=args.positive_threshold,
                    negative_threshold=args.negative_threshold,
                    meta=meta,
                )
            except Exception as exc:
                last_error = exc
                if is_sensitive_filter_error(exc) and caption_safety_mode != entry_variants[-1][0]:
                    logger.warning(
                        'Switching %s to metadata-only prompt after sensitive filter: %s',
                        h5_key,
                        exc,
                    )
                    break
                if attempt < int(args.max_retries):
                    logger.warning(
                        'Retrying %s mode=%s after attempt %d/%d failed: %s',
                        h5_key,
                        caption_safety_mode,
                        attempt + 1,
                        int(args.max_retries) + 1,
                        exc,
                    )
                    time.sleep(float(args.retry_wait_seconds))
    raise RuntimeError(f'Failed to build preference teacher for {h5_key}: {last_error}')


def main() -> None:
    args = get_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(levelname)s:%(name)s:%(message)s',
    )

    if args.num_perspectives <= 0 or args.num_perspectives > len(CANONICAL_PERSPECTIVE_IDS):
        raise ValueError(
            f'Invalid num_perspectives={args.num_perspectives}; '
            f'expected 1..{len(CANONICAL_PERSPECTIVE_IDS)}'
        )
    perspective_ids = CANONICAL_PERSPECTIVE_IDS[: int(args.num_perspectives)]

    structured = load_json(Path(args.structured_caption_path))
    output = Path(args.output)
    failures_output = Path(args.failures_output) if args.failures_output else output.with_suffix('.failures.json')
    records = {} if args.overwrite else load_existing_records(output)
    failures: Dict[str, Dict] = {}

    client = make_client(args)
    only_keys = set(args.only_keys or [])

    with h5py.File(args.h5_path, 'r') as h5:
        keys = sorted(h5.keys(), key=natural_key)
        if only_keys:
            keys = [key for key in keys if key in only_keys]
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError(f'Invalid limit={args.limit}; expected > 0.')
            keys = keys[: int(args.limit)]

        if not keys:
            raise ValueError('No HDF5 keys selected for teacher generation.')

        for index, h5_key in enumerate(keys, start=1):
            if h5_key in records and not args.overwrite:
                logger.info('[%d/%d] skip existing %s', index, len(keys), h5_key)
                continue
            if h5_key not in structured:
                failures[h5_key] = {'error': 'missing structured caption entry'}
                save_failures(failures_output, failures)
                continue
            logger.info('[%d/%d] building %s', index, len(keys), h5_key)
            try:
                records[h5_key] = process_video(
                    client=client,
                    args=args,
                    h5_key=h5_key,
                    h5_group=h5[h5_key],
                    structured_entry=structured[h5_key],
                    perspective_ids=perspective_ids,
                )
                save_records(output, records)
            except Exception as exc:
                logger.exception('Failed %s: %s', h5_key, exc)
                failures[h5_key] = {'error': repr(exc)}
                save_failures(failures_output, failures)

    save_records(output, records)
    save_failures(failures_output, failures)
    logger.info('Done | records=%d | failures=%d | output=%s', len(records), len(failures), output)
    if failures:
        raise RuntimeError(f'Preference teacher generation finished with failures: {failures_output}')


if __name__ == '__main__':
    main()
