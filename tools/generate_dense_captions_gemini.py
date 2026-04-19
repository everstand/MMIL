import sys
import os
import time
import json
import re
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.dataset_registry import get_pseudo_label_adapter


DEFAULT_MODEL_NAME = "gemini-3-flash-preview"
DEFAULT_BASE_URL = "https://us.novaiapi.com/v1"
DEFAULT_API_KEY_ENV = "NOVAI_API_KEY"

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}$")

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
]


SYSTEM_PROMPT = """You generate dense video captions for video summarization research.

Return only valid JSON.
Do not write markdown.
Do not write explanations.

Describe only visually observable content.
Do not infer hidden intentions, emotions, causes, or off-screen events.
Do not use abstract summary-style words such as highlight, important, key moment, climax, representative, informative, summary, salient, notable, memorable, crucial, or significant.

Generate 10 to 15 temporally ordered captions.
Use MM:SS timestamps.
Each caption must be one concise English sentence.

The captions should cover the video from beginning to end as evenly as possible.
Avoid near-duplicate captions for adjacent time ranges.
Prefer concrete actions, objects, scene changes, and visible interactions.

Every caption object must use exactly these three keys:
- start_mmss
- end_mmss
- caption

No extra keys are allowed.
Never use keys such as label, title, tag, description, note, or any other extra field.
For every caption object, end_mmss must be strictly later than start_mmss.
"""


USER_PROMPT = """Generate dense captions for this video.

Requirements:
1. Output JSON with one field only:
   - captions: array of objects
2. Each caption object must contain exactly these three keys and no others:
   - start_mmss
   - end_mmss
   - caption
3. Never use keys such as label, title, tag, description, note, or any extra field.
4. Captions must be temporally ordered.
5. For every caption object, end_mmss must be strictly later than start_mmss.
6. Captions must describe only visible actions, objects, scene changes, and interactions.
7. Do not infer invisible information or hidden intentions.
8. Do not write abstract summary statements.
9. Produce 10 to 15 captions.
10. Distribute captions across the video timeline instead of concentrating them in only one part.
11. Avoid repeated or nearly identical captions.
12. Use the frame timestamps provided below as temporal anchors.
"""


RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "captions": {
            "type": "array",
            "minItems": 10,
            "maxItems": 15,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_mmss": {
                        "type": "string",
                        "pattern": "^[0-9]{2}:[0-9]{2}$"
                    },
                    "end_mmss": {
                        "type": "string",
                        "pattern": "^[0-9]{2}:[0-9]{2}$"
                    },
                    "caption": {
                        "type": "string"
                    },
                },
                "required": ["start_mmss", "end_mmss", "caption"],
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

    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-side", type=int, default=960)

    parser.add_argument("--only-keys", nargs="*", default=None)

    return parser


def mmss_to_seconds(ts: str) -> int:
    if not isinstance(ts, str) or not TIMESTAMP_RE.fullmatch(ts):
        raise ValueError(f"Invalid MM:SS timestamp: {ts}")
    mm, ss = ts.split(":")
    mm_i = int(mm)
    ss_i = int(ss)
    if ss_i >= 60:
        raise ValueError(f"Invalid seconds field in timestamp: {ts}")
    return mm_i * 60 + ss_i


def seconds_to_mmss(seconds: float) -> str:
    seconds_int = max(0, int(round(seconds)))
    mm = seconds_int // 60
    ss = seconds_int % 60
    return f"{mm:02d}:{ss:02d}"


def normalize_caption_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text

def sanitize_caption_list(captions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(captions, list):
        raise ValueError(f"Expected captions to be a list, got {type(captions).__name__}")

    cleaned = []
    required_keys = ("start_mmss", "end_mmss", "caption")

    for idx, item in enumerate(captions):
        if not isinstance(item, dict):
            raise ValueError(f"Caption item {idx} is not an object")

        missing = [k for k in required_keys if k not in item]
        if missing:
            raise ValueError(f"Caption item {idx} missing required keys: {missing}")

        cleaned.append({
            "start_mmss": item["start_mmss"],
            "end_mmss": item["end_mmss"],
            "caption": item["caption"],
        })

    return cleaned

def validate_caption_list(captions: List[Dict[str, Any]]) -> None:
    if not isinstance(captions, list):
        raise ValueError(f"Expected captions to be a list, got {type(captions).__name__}")

    if not (10 <= len(captions) <= 15):
        raise ValueError(f"Expected 10-15 captions, got {len(captions)}")

    seen_caption_texts = set()
    prev_start = -1
    prev_end = -1

    for idx, item in enumerate(captions):
        if not isinstance(item, dict):
            raise ValueError(f"Caption item {idx} is not an object")

        expected_keys = {"start_mmss", "end_mmss", "caption"}
        actual_keys = set(item.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"Caption item {idx} has invalid keys: {sorted(actual_keys)}; "
                f"expected exactly {sorted(expected_keys)}"
            )

        start_mmss = item["start_mmss"]
        end_mmss = item["end_mmss"]
        caption = item["caption"]

        if not isinstance(start_mmss, str) or not start_mmss.strip():
            raise ValueError(f"Caption item {idx} invalid start_mmss")
        if not isinstance(end_mmss, str) or not end_mmss.strip():
            raise ValueError(f"Caption item {idx} invalid end_mmss")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Caption item {idx} invalid caption")

        start_sec = mmss_to_seconds(start_mmss)
        end_sec = mmss_to_seconds(end_mmss)

        if start_sec >= end_sec:
            raise ValueError(
                f"Caption item {idx} has non-positive interval: "
                f"{start_mmss} -> {end_mmss}"
            )

        if idx > 0:
            if start_sec < prev_start:
                raise ValueError(
                    f"Caption item {idx} start time is not temporally ordered: "
                    f"{start_mmss} comes after previous start."
                )
            if end_sec < prev_end:
                raise ValueError(
                    f"Caption item {idx} end time is not temporally ordered: "
                    f"{end_mmss} comes after previous end."
                )

        caption_norm = normalize_caption_text(caption)
        if caption_norm in seen_caption_texts:
            raise ValueError(f"Caption item {idx} is a duplicate caption: {caption}")
        seen_caption_texts.add(caption_norm)

        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(caption):
                raise ValueError(
                    f"Caption item {idx} contains forbidden summary-style phrase: "
                    f"{pattern.pattern}"
                )

        prev_start = start_sec
        prev_end = end_sec


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


def sample_video_frames(
    video_path: str,
    num_frames: int,
    jpeg_quality: int,
    max_side: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Invalid frame count for video: {video_path}")

    if fps <= 0:
        cap.release()
        raise ValueError(f"Invalid fps for video: {video_path}")

    frame_indices = sorted(set(
        int(round(i * (total_frames - 1) / max(num_frames - 1, 1)))
        for i in range(num_frames)
    ))

    sampled_frames: List[Dict[str, str]] = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = resize_frame_keep_aspect(frame, max_side=max_side)

        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            continue

        ts_sec = frame_idx / fps
        ts_mmss = seconds_to_mmss(ts_sec)

        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        sampled_frames.append({
            "frame_idx": int(frame_idx),
            "timestamp_mmss": ts_mmss,
            "data_url": f"data:image/jpeg;base64,{image_b64}",
        })

    cap.release()

    if not sampled_frames:
        raise ValueError(f"No valid sampled frames from video: {video_path}")

    meta = {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "sampled_count": len(sampled_frames),
    }
    return sampled_frames, meta


def build_user_content_from_frames(sampled_frames: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []

    content.append({"type": "text", "text": USER_PROMPT})
    content.append({
        "type": "text",
        "text": (
            "Below are uniformly sampled frames in temporal order. "
            "Each frame is paired with a timestamp in MM:SS. "
            "Use them as temporal anchors when generating captions."
        )
    })

    for item in sampled_frames:
        content.append({
            "type": "text",
            "text": f"Frame timestamp: {item['timestamp_mmss']}"
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
):
    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        msg = str(exc).lower()
        if ("response_format" in msg) or ("json_object" in msg) or ("unsupported" in msg):
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )
        raise


def generate_dense_captions_for_video(
    client: OpenAI,
    model_name: str,
    video_path: str,
    h5_key: str,
    max_retries: int,
    retry_wait_seconds: float,
    num_frames: int,
    jpeg_quality: int,
    max_side: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            sampled_frames, sample_meta = sample_video_frames(
                video_path=video_path,
                num_frames=num_frames,
                jpeg_quality=jpeg_quality,
                max_side=max_side,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_content_from_frames(sampled_frames)},
            ]

            response = request_chat_completion_once(
                client=client,
                model_name=model_name,
                messages=messages,
            )

            text = extract_response_text(response)
            obj = json.loads(text)

            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object response, got {type(obj).__name__}")

            if "captions" not in obj:
                raise ValueError('Response JSON missing "captions" field.')

            captions = sanitize_caption_list(obj["captions"])
            validate_caption_list(captions)
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


def main() -> None:
    args = get_parser().parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env: {args.api_key_env}")

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
    )

    adapter = get_pseudo_label_adapter(args.dataset)
    items = adapter.resolve_items(
        video_dir=args.video_dir,
        h5_path=args.h5_path,
    )

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
    for item in failure_list:
        if isinstance(item, dict) and "h5_key" in item:
            failures_by_key[item["h5_key"]] = item

    total_items = len(items)
    completed_count = 0

    for idx, item in enumerate(items, start=1):
        h5_key = item["h5_key"]
        raw_video_name = item.get("raw_video_name", None)
        video_path = str(item["video_path"])

        if h5_key in structured_result and h5_key in simple_result:
            completed_count += 1
            print(f"[{idx}/{total_items}] Skip {h5_key} | already exists", flush=True)
            continue

        print(f"[{idx}/{total_items}] Start {h5_key} | {video_path}", flush=True)

        try:
            captions, sample_meta = generate_dense_captions_for_video(
                client=client,
                model_name=args.model_name,
                video_path=video_path,
                h5_key=h5_key,
                max_retries=args.max_retries,
                retry_wait_seconds=args.retry_wait_seconds,
                num_frames=args.num_frames,
                jpeg_quality=args.jpeg_quality,
                max_side=args.max_side,
            )

            structured_result[h5_key] = {
                "sample_meta": sample_meta,
                "captions": captions,
            }
            simple_result[h5_key] = [x["caption"] for x in captions]

            if h5_key in failures_by_key:
                failures_by_key.pop(h5_key)

            completed_count += 1
            print(
                f"[{idx}/{total_items}] Done {h5_key} | captions={len(captions)} | sampled_frames={sample_meta['sampled_count']}",
                flush=True,
            )

        except Exception as exc:
            failures_by_key[h5_key] = {
                "h5_key": h5_key,
                "raw_video_name": "" if raw_video_name is None else str(raw_video_name),
                "video_path": video_path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            print(f"[{idx}/{total_items}] Failed {h5_key} | {type(exc).__name__}: {exc}", flush=True)

        save_outputs(
            out_structured=out_structured,
            out_simple=out_simple,
            structured_result=structured_result,
            simple_result=simple_result,
            failures_by_key=failures_by_key,
        )

        time.sleep(args.sleep_seconds)

    print(f"[Done] structured={out_structured}", flush=True)
    print(f"[Done] simple={out_simple}", flush=True)
    print(f"[Done] failures={failure_path}", flush=True)
    print(f"[Done] completed={completed_count}/{total_items}", flush=True)


if __name__ == "__main__":
    main()