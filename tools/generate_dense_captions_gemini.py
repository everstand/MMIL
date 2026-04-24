import sys
import os
import time
import json
import re
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.dataset_registry import get_pseudo_label_adapter


DEFAULT_MODEL_NAME = "gemini-3-flash-preview"
DEFAULT_BASE_URL = "https://api.xheai.cc/v1/"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

FORBIDDEN_PATTERNS = [
    re.compile(r"\bhighlight\b", re.IGNORECASE),
    re.compile(r"\bimportant\b", re.IGNORECASE),
    re.compile(r"\bkey\s+moment\b", re.IGNORECASE),
    re.compile(r"\bclimax\b", re.IGNORECASE),
    re.compile(r"\brepresentative\b", re.IGNORECASE),
    re.compile(r"\binformative\b", re.IGNORECASE),
    re.compile(r"\bsummary\b", re.IGNORECASE),
    re.compile(r"\bsalient\b", re.IGNORECASE),
    re.compile(r"\bnotable\b", re.IGNORECASE),
    re.compile(r"\bmemorable\b", re.IGNORECASE),
    re.compile(r"\bcrucial\b", re.IGNORECASE),
    re.compile(r"\bsignificant\b", re.IGNORECASE),
    re.compile(r"\bhigh\s+information\b", re.IGNORECASE),
    re.compile(r"\bexciting\b", re.IGNORECASE),
    re.compile(r"\binteresting\b", re.IGNORECASE),
    re.compile(r"\bboring\b", re.IGNORECASE),
    re.compile(r"\bfocus\s+on\b", re.IGNORECASE),
]

SYSTEM_PROMPT = """You are a cold visual event logger for video summarization research.

Return only valid JSON.
Do not write markdown.
Do not write explanations.

Describe only visually observable content in the provided sampled frames.
Do not infer hidden intentions, emotions, causes, importance, relevance, or off-screen events.
Do not use abstract summary-style or saliency-style words.
Do not create new wording just to avoid repetition.
If adjacent sampled frames show the same visible state or the same repeated action, merge them into one longer segment. If separate adjacent segments are unavoidable and the visible content is unchanged, reusing the same factual caption is allowed.

The output must be temporally ordered.
Each caption must be one concise English sentence.
Every caption object must use exactly these three keys:
- start_frame_id
- end_frame_id
- caption

No extra keys are allowed.
start_frame_id and end_frame_id must refer to frame IDs explicitly provided in the input.
end_frame_id must be strictly larger than start_frame_id.
"""


def build_user_prompt(min_captions: int, max_captions: int) -> str:
    return f"""Generate factual visual segments for this video from the sampled frames.

Requirements:
1. Output JSON with one field only:
   - captions: array of objects
2. Each caption object must contain exactly these three keys and no others:
   - start_frame_id: integer
   - end_frame_id: integer
   - caption: string
3. start_frame_id and end_frame_id must be chosen from the frame IDs listed below.
4. end_frame_id must be strictly larger than start_frame_id.
5. Use only the sampled frames and the visible content in them.
6. Do not infer invisible information, emotions, intention, importance, or summary value.
7. Use the fewest segments that faithfully describe visible changes.
8. Do not force segmentation when the scene or action remains visually redundant.
9. If there are only a few visual stages, output only a few captions.
10. If there are many visible changes, output more captions, up to the maximum.
11. Output between {min_captions} and {max_captions} captions.
12. Merge consecutive visually unchanged frames into one segment.
13. Do not paraphrase repeated content merely to make adjacent captions look different.
"""


def build_response_json_schema(min_captions: int, max_captions: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "captions": {
                "type": "array",
                "minItems": int(min_captions),
                "maxItems": int(max_captions),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_frame_id": {"type": "integer", "minimum": 0},
                        "end_frame_id": {"type": "integer", "minimum": 0},
                        "caption": {"type": "string"},
                    },
                    "required": ["start_frame_id", "end_frame_id", "caption"],
                },
            },
        },
        "required": ["captions"],
    }


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=("summe", "tvsum"))
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--h5-path", type=str, required=True)

    parser.add_argument("--out-structured", type=str, required=True)
    parser.add_argument("--out-simple", type=str, required=True)

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-wait-seconds", type=float, default=5.0)

    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--min-captions", type=int, default=1)
    parser.add_argument("--max-captions", type=int, default=24)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--only-keys", nargs="*", default=None)

    return parser


def seconds_to_mmss_floor(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    mm = seconds_int // 60
    ss = seconds_int % 60
    return f"{mm:02d}:{ss:02d}"


def seconds_to_mmss_ceil(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    if float(seconds_int) < float(seconds):
        seconds_int += 1
    mm = seconds_int // 60
    ss = seconds_int % 60
    return f"{mm:02d}:{ss:02d}"


def mmss_to_seconds(ts: str) -> int:
    if not isinstance(ts, str) or not re.fullmatch(r"\d{2}:\d{2}", ts):
        raise ValueError(f"Invalid MM:SS timestamp: {ts}")
    mm, ss = ts.split(":")
    mm_i = int(mm)
    ss_i = int(ss)
    if ss_i >= 60:
        raise ValueError(f"Invalid seconds field in timestamp: {ts}")
    return mm_i * 60 + ss_i


def normalize_caption_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def load_existing_json(path: Path, default_value):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_value


def save_outputs(
    out_structured: Path,
    out_simple: Path,
    structured_result: Dict[str, Any],
    simple_result: Dict[str, List[str]],
    failures_by_key: Dict[str, Dict[str, str]],
) -> None:
    with open(out_structured, "w", encoding="utf-8") as f:
        json.dump(structured_result, f, indent=2, ensure_ascii=False)

    with open(out_simple, "w", encoding="utf-8") as f:
        json.dump(simple_result, f, indent=2, ensure_ascii=False)

    failure_path = out_structured.with_suffix(".failures.json")
    failure_list = [failures_by_key[k] for k in sorted(failures_by_key.keys())]
    with open(failure_path, "w", encoding="utf-8") as f:
        json.dump(failure_list, f, indent=2, ensure_ascii=False)


def resize_frame_keep_aspect(frame, max_side: int):
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame

    scale = max_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_frame_as_data_url(frame, jpeg_quality: int, max_side: int) -> str:
    frame = resize_frame_keep_aspect(frame, max_side=max_side)
    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("Failed to JPEG-encode sampled frame.")
    image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"


def sample_video_frames_sequential(
    video_path: str,
    num_frames: int,
    jpeg_quality: int,
    max_side: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")

    # Pass 1: audit real decodable frame count.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid fps for video: {video_path}")

    actual_total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        actual_total_frames += 1

    cap.release()

    if actual_total_frames <= 1:
        raise ValueError(
            f"Invalid actual decodable frame count for video: {video_path}, "
            f"actual_total_frames={actual_total_frames}, "
            f"reported_total_frames={reported_total_frames}"
        )

    effective_num_frames = min(int(num_frames), int(actual_total_frames))

    target_indices = sorted(set(
        int(round(i * (actual_total_frames - 1) / max(effective_num_frames - 1, 1)))
        for i in range(effective_num_frames)
    ))

    target_set = set(target_indices)
    last_target = target_indices[-1]

    # Pass 2: decode selected frames using actual decodable frame indices.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot reopen video: {video_path}")

    sampled_by_frame_idx: Dict[int, Dict[str, Any]] = {}
    current_idx = 0

    while current_idx <= last_target:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if current_idx in target_set:
            ts_sec = current_idx / fps
            sampled_by_frame_idx[current_idx] = {
                "frame_idx": int(current_idx),
                "timestamp_sec": float(ts_sec),
                "timestamp_mmss": seconds_to_mmss_floor(ts_sec),
                "data_url": encode_frame_as_data_url(
                    frame=frame,
                    jpeg_quality=jpeg_quality,
                    max_side=max_side,
                ),
            }

        current_idx += 1

    cap.release()

    missing = [idx for idx in target_indices if idx not in sampled_by_frame_idx]
    if missing:
        raise RuntimeError(
            f"Failed to sequentially decode sampled frames from {video_path} "
            f"even after actual-frame audit. "
            f"decoded={len(sampled_by_frame_idx)}/{len(target_indices)}, "
            f"first_missing_frame={missing[0]}, "
            f"last_scanned_frame={current_idx - 1}, "
            f"actual_total_frames={actual_total_frames}, "
            f"reported_total_frames={reported_total_frames}"
        )

    sampled_frames: List[Dict[str, Any]] = []
    for frame_id, frame_idx in enumerate(target_indices):
        record = dict(sampled_by_frame_idx[frame_idx])
        record["frame_id"] = int(frame_id)
        sampled_frames.append(record)

    if len(sampled_frames) < 2:
        raise ValueError(
            f"Need at least 2 sampled frames, got {len(sampled_frames)} for {video_path}"
        )

    meta = {
        "reported_total_frames": int(reported_total_frames),
        "actual_total_frames": int(actual_total_frames),
        "total_frames": int(actual_total_frames),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "requested_sampled_count": int(num_frames),
        "sampled_count": int(len(sampled_frames)),
        "sampling_backend": "opencv_actual_decodable_two_pass",
        "sampled_frames": [
            {
                "frame_id": int(x["frame_id"]),
                "frame_idx": int(x["frame_idx"]),
                "timestamp_sec": float(x["timestamp_sec"]),
                "timestamp_mmss": str(x["timestamp_mmss"]),
            }
            for x in sampled_frames
        ],
    }

    return sampled_frames, meta


def build_user_content_from_frames(
    sampled_frames: List[Dict[str, Any]],
    min_captions: int,
    max_captions: int,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": build_user_prompt(min_captions, max_captions)})
    content.append({
        "type": "text",
        "text": (
            "Below are uniformly sampled frames in temporal order. "
            "Each frame has a frame_id and timestamp. Return only frame_id ranges. "
            "The timestamp is provided for context only; do not output timestamps."
        )
    })

    for item in sampled_frames:
        content.append({
            "type": "text",
            "text": (
                f"Frame ID: {item['frame_id']} | "
                f"Video frame index: {item['frame_idx']} | "
                f"Timestamp: {item['timestamp_mmss']}"
            )
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": item["data_url"]}
        })

    return content


def extract_response_text(response) -> str:
    if not hasattr(response, "choices") or not response.choices:
        raise ValueError("Model response does not contain choices.")

    message = response.choices[0].message
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        text = "\n".join([x for x in text_parts if x])
        if text:
            return text

    raise ValueError("Unable to extract text content from model response.")


def request_chat_completion_once(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    schema: Dict[str, Any],
    temperature: float,
):
    json_schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "dense_caption_segments",
            "strict": True,
            "schema": schema,
        },
    }

    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format=json_schema_format,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if ("json_schema" not in msg) and ("response_format" not in msg) and ("unsupported" not in msg):
            raise

    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        msg = str(exc).lower()
        if ("json_object" not in msg) and ("response_format" not in msg) and ("unsupported" not in msg):
            raise

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )


def parse_json_response_strict(text: str) -> Dict[str, Any]:
    # Do not strip markdown fences. If the endpoint returns markdown, structured output failed.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Model response is not strict JSON. Do not regex-clean markdown fences; retry or fix response_format support."
        ) from exc

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object response, got {type(obj).__name__}")
    return obj


def validate_model_segments(
    segments: Any,
    sampled_frames: List[Dict[str, Any]],
    min_captions: int,
    max_captions: int,
) -> List[Dict[str, Any]]:
    if not isinstance(segments, list):
        raise ValueError(f"Expected captions to be a list, got {type(segments).__name__}")

    if not (min_captions <= len(segments) <= max_captions):
        raise ValueError(
            f"Expected {min_captions}-{max_captions} captions, got {len(segments)}"
        )

    max_frame_id = len(sampled_frames) - 1
    cleaned: List[Dict[str, Any]] = []
    prev_start = -1
    prev_end = -1

    for idx, item in enumerate(segments):
        if not isinstance(item, dict):
            raise ValueError(f"Caption item {idx} is not an object")

        expected_keys = {"start_frame_id", "end_frame_id", "caption"}
        actual_keys = set(item.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"Caption item {idx} has invalid keys: {sorted(actual_keys)}; "
                f"expected exactly {sorted(expected_keys)}"
            )

        start_frame_id = int(item["start_frame_id"])
        end_frame_id = int(item["end_frame_id"])
        caption = str(item["caption"]).strip()

        if not caption:
            raise ValueError(f"Caption item {idx} has empty caption")
        if start_frame_id < 0 or start_frame_id > max_frame_id:
            raise ValueError(f"Caption item {idx} invalid start_frame_id={start_frame_id}")
        if end_frame_id < 0 or end_frame_id > max_frame_id:
            raise ValueError(f"Caption item {idx} invalid end_frame_id={end_frame_id}")
        if end_frame_id <= start_frame_id:
            raise ValueError(
                f"Caption item {idx} must satisfy end_frame_id > start_frame_id, "
                f"got {start_frame_id}->{end_frame_id}"
            )
        if idx > 0:
            if start_frame_id < prev_start:
                raise ValueError(
                    f"Caption item {idx} start_frame_id is not temporally ordered: "
                    f"{start_frame_id} after previous {prev_start}"
                )
            if end_frame_id < prev_end:
                raise ValueError(
                    f"Caption item {idx} end_frame_id is not temporally ordered: "
                    f"{end_frame_id} after previous {prev_end}"
                )

        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(caption):
                raise ValueError(
                    f"Caption item {idx} contains forbidden summary-style phrase: {pattern.pattern}"
                )

        cleaned.append({
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "caption": caption,
        })
        prev_start = start_frame_id
        prev_end = end_frame_id

    return merge_adjacent_identical_segments(cleaned)


def merge_adjacent_identical_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        same_caption = normalize_caption_text(prev["caption"]) == normalize_caption_text(seg["caption"])
        adjacent_or_overlapping = int(seg["start_frame_id"]) <= int(prev["end_frame_id"]) + 1
        if same_caption and adjacent_or_overlapping:
            prev["end_frame_id"] = max(int(prev["end_frame_id"]), int(seg["end_frame_id"]))
        else:
            merged.append(dict(seg))
    return merged


def convert_frame_segments_to_mmss_captions(
    segments: List[Dict[str, Any]],
    sampled_frames: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    captions: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        start_frame_id = int(seg["start_frame_id"])
        end_frame_id = int(seg["end_frame_id"])
        start_sec = float(sampled_frames[start_frame_id]["timestamp_sec"])
        end_sec = float(sampled_frames[end_frame_id]["timestamp_sec"])

        start_mmss = seconds_to_mmss_floor(start_sec)
        end_mmss = seconds_to_mmss_ceil(end_sec)

        if mmss_to_seconds(end_mmss) <= mmss_to_seconds(start_mmss):
            end_mmss = seconds_to_mmss_ceil(start_sec + 1.0)

        captions.append({
            "start_mmss": start_mmss,
            "end_mmss": end_mmss,
            "caption": str(seg["caption"]).strip(),
        })

    validate_final_caption_list(captions)
    return captions


def validate_final_caption_list(captions: List[Dict[str, Any]]) -> None:
    if not captions:
        raise ValueError("No final captions produced.")

    prev_start = -1
    prev_end = -1
    for idx, item in enumerate(captions):
        if set(item.keys()) != {"start_mmss", "end_mmss", "caption"}:
            raise ValueError(f"Final caption {idx} has invalid keys: {sorted(item.keys())}")

        start_sec = mmss_to_seconds(item["start_mmss"])
        end_sec = mmss_to_seconds(item["end_mmss"])
        if end_sec <= start_sec:
            raise ValueError(
                f"Final caption {idx} has non-positive interval: "
                f"{item['start_mmss']}->{item['end_mmss']}"
            )
        if idx > 0:
            if start_sec < prev_start:
                raise ValueError(f"Final caption {idx} start time is not ordered.")
            if end_sec < prev_end:
                raise ValueError(f"Final caption {idx} end time is not ordered.")

        caption = str(item["caption"]).strip()
        if not caption:
            raise ValueError(f"Final caption {idx} is empty.")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(caption):
                raise ValueError(
                    f"Final caption {idx} contains forbidden summary-style phrase: {pattern.pattern}"
                )

        prev_start = start_sec
        prev_end = end_sec


def generate_dense_captions_for_video(
    client: OpenAI,
    model_name: str,
    video_path: str,
    h5_key: str,
    max_retries: int,
    retry_wait_seconds: float,
    num_frames: int,
    min_captions: int,
    max_captions: int,
    jpeg_quality: int,
    max_side: int,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            sampled_frames, sample_meta = sample_video_frames_sequential(
                video_path=video_path,
                num_frames=num_frames,
                jpeg_quality=jpeg_quality,
                max_side=max_side,
            )

            max_segments_by_frames = max(1, len(sampled_frames) - 1)
            effective_max_captions = min(int(max_captions), max_segments_by_frames)
            effective_min_captions = min(int(min_captions), effective_max_captions)
            effective_min_captions = max(1, effective_min_captions)

            schema = build_response_json_schema(effective_min_captions, effective_max_captions)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_content_from_frames(
                        sampled_frames=sampled_frames,
                        min_captions=effective_min_captions,
                        max_captions=effective_max_captions,
                    ),
                },
            ]

            response = request_chat_completion_once(
                client=client,
                model_name=model_name,
                messages=messages,
                schema=schema,
                temperature=temperature,
            )

            text = extract_response_text(response)
            obj = parse_json_response_strict(text)

            if "captions" not in obj:
                raise ValueError('Response JSON missing "captions" field.')

            frame_segments = validate_model_segments(
                segments=obj["captions"],
                sampled_frames=sampled_frames,
                min_captions=effective_min_captions,
                max_captions=effective_max_captions,
            )
            captions = convert_frame_segments_to_mmss_captions(
                segments=frame_segments,
                sampled_frames=sampled_frames,
            )

            sample_meta["requested_min_captions"] = int(min_captions)
            sample_meta["requested_max_captions"] = int(max_captions)
            sample_meta["effective_min_captions"] = int(effective_min_captions)
            sample_meta["effective_max_captions"] = int(effective_max_captions)
            sample_meta["model_output_segments"] = frame_segments
            return captions, sample_meta

        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                print(
                    f"      Retry {attempt}/{max_retries} failed for {h5_key}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(retry_wait_seconds)
            else:
                raise

    raise RuntimeError(f"Unexpected retry loop exit for {h5_key}: {last_error}")


def process_item(
    item: Dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[List[str]], Optional[Dict[str, str]]]:
    h5_key = item["h5_key"]
    raw_video_name = item.get("raw_video_name", None)
    video_path = str(item["video_path"])

    try:
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        captions, sample_meta = generate_dense_captions_for_video(
            client=client,
            model_name=args.model_name,
            video_path=video_path,
            h5_key=h5_key,
            max_retries=args.max_retries,
            retry_wait_seconds=args.retry_wait_seconds,
            num_frames=args.num_frames,
            min_captions=args.min_captions,
            max_captions=args.max_captions,
            jpeg_quality=args.jpeg_quality,
            max_side=args.max_side,
            temperature=args.temperature,
        )
        structured_entry = {
            "sample_meta": sample_meta,
            "captions": captions,
        }
        simple_entry = [x["caption"] for x in captions]
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        return h5_key, structured_entry, simple_entry, None
    except Exception as exc:
        failure = {
            "h5_key": h5_key,
            "raw_video_name": "" if raw_video_name is None else str(raw_video_name),
            "video_path": video_path,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        return h5_key, None, None, failure


def main() -> None:
    args = get_parser().parse_args()

    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}")
    if args.num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {args.num_frames}")
    if args.min_captions <= 0:
        raise ValueError(f"min_captions must be positive, got {args.min_captions}")
    if args.max_captions < args.min_captions:
        raise ValueError(
            f"max_captions must be >= min_captions, got {args.max_captions} < {args.min_captions}"
        )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env: {args.api_key_env}")

    adapter = get_pseudo_label_adapter(args.dataset)
    items = adapter.resolve_items(video_dir=args.video_dir, h5_path=args.h5_path)

    if args.only_keys is not None and len(args.only_keys) > 0:
        only_keys_set = set(args.only_keys)
        items = [item for item in items if item["h5_key"] in only_keys_set]

    if args.limit is not None:
        items = items[:args.limit]

    out_structured = Path(args.out_structured)
    out_simple = Path(args.out_simple)
    out_structured.parent.mkdir(parents=True, exist_ok=True)
    out_simple.parent.mkdir(parents=True, exist_ok=True)

    structured_result = load_existing_json(out_structured, {})
    simple_result = load_existing_json(out_simple, {})

    failure_path = out_structured.with_suffix(".failures.json")
    failure_list = load_existing_json(failure_path, [])
    failures_by_key: Dict[str, Dict[str, str]] = {}
    for failure_item in failure_list:
        if isinstance(failure_item, dict) and "h5_key" in failure_item:
            failures_by_key[str(failure_item["h5_key"])] = failure_item

    pending_items = []
    total_items = len(items)
    completed_count = 0

    for idx, item in enumerate(items, start=1):
        h5_key = item["h5_key"]
        if h5_key in structured_result and h5_key in simple_result:
            completed_count += 1
            print(f"[{idx}/{total_items}] skip | {h5_key}", flush=True)
        else:
            pending_items.append((idx, item))

    if not pending_items:
        save_outputs(out_structured, out_simple, structured_result, simple_result, failures_by_key)
        print(f"[Done] structured={out_structured}", flush=True)
        print(f"[Done] simple={out_simple}", flush=True)
        print(f"[Done] failures={failure_path}", flush=True)
        print(f"[Done] completed={completed_count}/{total_items}", flush=True)
        return

    if args.num_workers == 1:
        for idx, item in pending_items:
            h5_key = item["h5_key"]
            print(f"[{idx}/{total_items}] start | {h5_key}", flush=True)
            key, structured_entry, simple_entry, failure = process_item(item, args, api_key)

            if failure is None:
                structured_result[key] = structured_entry
                simple_result[key] = simple_entry
                failures_by_key.pop(key, None)
                completed_count += 1
                print(
                    f"[{idx}/{total_items}] done | {key} | captions={len(structured_entry['captions'])}",
                    flush=True,
                )
            else:
                failures_by_key[key] = failure
                print(
                    f"[{idx}/{total_items}] failed | {key} | {failure['error_type']}: {failure['error_message']}",
                    flush=True,
                )

            save_outputs(out_structured, out_simple, structured_result, simple_result, failures_by_key)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_item, item, args, api_key): (idx, item)
                for idx, item in pending_items
            }

            for future in as_completed(futures):
                idx, item = futures[future]
                h5_key = item["h5_key"]
                try:
                    key, structured_entry, simple_entry, failure = future.result()
                except Exception as exc:
                    key = h5_key
                    failure = {
                        "h5_key": h5_key,
                        "raw_video_name": "" if item.get("raw_video_name") is None else str(item.get("raw_video_name")),
                        "video_path": str(item["video_path"]),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    structured_entry = None
                    simple_entry = None

                if failure is None:
                    structured_result[key] = structured_entry
                    simple_result[key] = simple_entry
                    failures_by_key.pop(key, None)
                    completed_count += 1
                    print(
                        f"[{idx}/{total_items}] done | {key} | captions={len(structured_entry['captions'])}",
                        flush=True,
                    )
                else:
                    failures_by_key[key] = failure
                    print(
                        f"[{idx}/{total_items}] failed | {key} | {failure['error_type']}: {failure['error_message']}",
                        flush=True,
                    )

                save_outputs(out_structured, out_simple, structured_result, simple_result, failures_by_key)

    print(f"[Done] structured={out_structured}", flush=True)
    print(f"[Done] simple={out_simple}", flush=True)
    print(f"[Done] failures={failure_path}", flush=True)
    print(f"[Done] completed={completed_count}/{total_items}", flush=True)


if __name__ == "__main__":
    main()
