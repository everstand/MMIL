#!/usr/bin/env python3
"""Closure-aware teacher-alone validation for SumMe.

This script reconstructs one internal non-human teacher candidate:
    value_v2 = phase1_default + alpha * closure_support - beta * unsupported_peak

Teacher construction uses only existing assets:
- structured dense captions
- text features
- H5 timing fields: change_points, n_frame_per_seg, picks
- existing shot_utility components

`gtscore` and `user_summary` are read only for teacher-alone validation gates.
No student training, checkpoint selection, or training pseudo-label construction is performed here.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_summe_objective_conflict import (
    build_summary_from_bitset,
    exact_oracle_max_user,
    normalize_gtscore,
    rank_or_nan,
    scalar_int,
)
from analyze_teacher_component_attribution import (
    bool_from_keyshot_summary,
    collect_unique_keys,
    f1_binary,
    load_shot_utility,
    open_group,
    precision_recall,
)
from helpers import vsumm_helper
from helpers.eval_protocol_helper import infer_f1_metric_from_dataset, safe_nanmean
from helpers.shot_utility_helper import build_components, normalize_01
from run_train_mil_cond import load_all_splits, validate_splits

CLOSURE_PATTERNS = [
    r'\bafter\b', r'\bfinally\b', r'\bend(?:s|ed|ing)?\b', r'\bfinish(?:es|ed|ing)?\b',
    r'\bcomplete(?:s|d)?\b', r'\bresult(?:s|ed)?\b', r'\boutcome\b', r'\bthen\b',
    r'\bnow\b', r'\bstands?\b', r'\bstops?\b', r'\bpauses?\b', r'\bsettles?\b',
    r'\brests?\b', r'\bremains?\b', r'\bleaves?\b', r'\bwalks away\b', r'\bturns away\b',
    r'\blooks back\b', r'\bshows? the result\b', r'\baftermath\b', r'\bopened\b',
    r'\bclosed\b', r'\bempty\b', r'\bclears?\b', r'\bcalm\b', r'\bstill\b',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Closure-aware teacher-alone validation.')
    parser.add_argument('--dataset', default='summe', choices=('summe',))
    parser.add_argument('--splits', nargs='+', default=['splits/summe.yml'])
    parser.add_argument('--selection-part', default='val', choices=('all', 'train', 'val', 'test'))
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=19500)
    parser.add_argument('--structured-caption-path', default='captions_raw/summe_dense_captions_structured_v3.json')
    parser.add_argument('--text-feature-path', default='features/text_summe_v3.h5')
    parser.add_argument('--shot-utility-path', default='pseudo_labels/summe/shot_utility_v3.npy')
    parser.add_argument('--summary-budget', type=float, default=0.15)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--peak-quantile', type=float, default=0.75)
    parser.add_argument('--lookahead', type=int, default=3)
    parser.add_argument('--min-continuity', type=float, default=0.35)
    parser.add_argument('--gate-f1', type=float, default=0.4541)
    parser.add_argument('--gate-tau', type=float, default=0.0311)
    parser.add_argument('--gate-rho', type=float, default=0.0472)
    parser.add_argument('--gate-recall', type=float, default=0.4619)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument('--output-json', default='diagnostics/summe_closure_aware_teacher_v2_val.json')
    parser.add_argument('--output-csv', default='diagnostics/summe_closure_aware_teacher_v2_val.csv')
    parser.add_argument('--teacher-output', default='')
    parser.add_argument('--force-write', action='store_true')
    return parser.parse_args()


class TextFeatureStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'Text feature file not found: {self.path}')
        self.handle = h5py.File(str(self.path), 'r')

    def close(self):
        self.handle.close()

    def get(self, h5_key: str) -> np.ndarray:
        if h5_key not in self.handle:
            raise KeyError(f'Missing text feature key {h5_key} in {self.path}')
        arr = self.handle[h5_key]['all_text_features'][...].astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f'Expected all_text_features [N,D], got {arr.shape} for {h5_key}')
        return arr


def load_structured_captions(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f'Structured caption file not found: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f'Structured caption file must contain a non-empty dict: {path}')
    return obj


def mmss_to_seconds(value: str) -> float:
    parts = str(value).strip().split(':')
    if len(parts) == 2:
        return float(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f'Invalid timestamp: {value}')


def cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return max(0.0, min(1.0, 0.5 * (float(np.dot(a, b) / denom) + 1.0)))


def closure_lexical_score(text: str) -> float:
    text_l = str(text).lower()
    hits = sum(1 for pattern in CLOSURE_PATTERNS if re.search(pattern, text_l))
    return 0.0 if hits <= 0 else float(min(1.0, 0.35 + 0.20 * hits))


def build_shot_caption_evidence(entry: Dict, text_features: np.ndarray, cps: np.ndarray, n_frames: int):
    captions = entry.get('captions')
    if not isinstance(captions, list) or not captions:
        raise ValueError('Structured caption entry missing captions list.')
    if len(captions) != text_features.shape[0]:
        raise ValueError(f'Caption/text count mismatch: {len(captions)} vs {text_features.shape[0]}')
    meta = entry.get('sample_meta', {})
    fps = float(meta.get('fps', 0.0))
    if fps <= 0:
        duration = float(meta.get('duration_sec', 0.0))
        fps = float(n_frames) / duration if duration > 0 else 1.0

    num_shots = int(cps.shape[0])
    feat_dim = int(text_features.shape[1])
    shot_features = np.zeros((num_shots, feat_dim), dtype=np.float32)
    shot_weights = np.zeros((num_shots,), dtype=np.float32)
    lexical = np.zeros((num_shots,), dtype=np.float32)
    mids = []
    for cap_idx, cap in enumerate(captions):
        start = int(round(mmss_to_seconds(cap.get('start_mmss', '0:00')) * fps))
        end = int(round(mmss_to_seconds(cap.get('end_mmss', cap.get('start_mmss', '0:00'))) * fps))
        if end <= start:
            end = start + 1
        mids.append(0.5 * (start + end))
        lex = closure_lexical_score(cap.get('caption', ''))
        for shot_idx, (first, last) in enumerate(cps):
            inter = min(end, int(last) + 1) - max(start, int(first))
            if inter > 0:
                shot_features[shot_idx] += float(inter) * text_features[cap_idx]
                shot_weights[shot_idx] += float(inter)
                lexical[shot_idx] = max(float(lexical[shot_idx]), lex)
    mids = np.asarray(mids, dtype=np.float32)
    for shot_idx, (first, last) in enumerate(cps):
        if shot_weights[shot_idx] > 0:
            shot_features[shot_idx] /= max(float(shot_weights[shot_idx]), 1e-6)
        else:
            center = 0.5 * (int(first) + int(last))
            nearest = int(np.argmin(np.abs(mids - center)))
            shot_features[shot_idx] = text_features[nearest]
            lexical[shot_idx] = closure_lexical_score(captions[nearest].get('caption', ''))
    return shot_features.astype(np.float32), lexical.astype(np.float32)


def build_closure_components(record: Dict, entry: Dict, text_features: np.ndarray, cps: np.ndarray, n_frames: int, args):
    comp = build_components(record)
    num_shots = int(cps.shape[0])
    for name in ['phase1_default', 'eventiveness', 'caption_change', 'visual_change', 'distinctiveness', 'anti_redundancy', 'semantic', 'representativeness']:
        value = comp.get(name)
        if not isinstance(value, np.ndarray) or value.shape[0] != num_shots:
            raise ValueError(f'Missing or invalid component: {name}')
    shot_text, closure_lex = build_shot_caption_evidence(entry, text_features, cps, n_frames)
    phase1 = normalize_01(comp['phase1_default'])
    semantic_base = normalize_01(0.55 * comp['semantic'] + 0.45 * comp['representativeness'])
    peak_score = normalize_01(
        0.35 * comp['eventiveness'] + 0.25 * comp['caption_change'] + 0.20 * comp['visual_change']
        + 0.10 * comp['distinctiveness'] + 0.10 * comp['anti_redundancy']
    )
    closure_support = np.zeros((num_shots,), dtype=np.float32)
    unsupported_peak = np.zeros((num_shots,), dtype=np.float32)
    max_future_closure = np.zeros((num_shots,), dtype=np.float32)
    if num_shots > 1:
        threshold = float(np.quantile(peak_score, float(args.peak_quantile)))
        for peak_idx in range(num_shots):
            if float(peak_score[peak_idx]) < threshold:
                continue
            best = 0.0
            for offset in range(1, int(args.lookahead) + 1):
                cand_idx = peak_idx + offset
                if cand_idx >= num_shots:
                    break
                continuity = cosine01(shot_text[peak_idx], shot_text[cand_idx])
                if continuity < float(args.min_continuity):
                    continue
                post_settle = max(0.0, float(peak_score[peak_idx] - peak_score[cand_idx]))
                closure_nature = min(1.0, max(0.0,
                    0.45 * float(closure_lex[cand_idx])
                    + 0.25 * float(semantic_base[cand_idx])
                    + 0.20 * post_settle
                    + 0.10 * float(phase1[cand_idx])
                ))
                evidence = float(peak_score[peak_idx]) * continuity * closure_nature / float(offset)
                closure_support[cand_idx] = max(float(closure_support[cand_idx]), evidence)
                best = max(best, evidence)
            max_future_closure[peak_idx] = best
            unsupported_peak[peak_idx] = float(peak_score[peak_idx]) * (1.0 - min(1.0, best))
    closure_support = normalize_01(closure_support)
    unsupported_peak = normalize_01(unsupported_peak)
    value_v2 = normalize_01(phase1 + float(args.alpha) * closure_support - float(args.beta) * unsupported_peak)
    return {
        'shot_scores': value_v2.astype(np.float32),
        'closure_support': closure_support.astype(np.float32),
        'unsupported_peak': unsupported_peak.astype(np.float32),
        'peak_score': peak_score.astype(np.float32),
        'closure_lexical': closure_lex.astype(np.float32),
        'semantic_base': semantic_base.astype(np.float32),
        'max_future_closure': max_future_closure.astype(np.float32),
    }


def project_shot_scores_to_sampled(shot_scores: np.ndarray, cps: np.ndarray, picks: np.ndarray) -> np.ndarray:
    out = np.zeros((picks.shape[0],), dtype=np.float32)
    for idx, frame_idx in enumerate(picks):
        hits = np.where((cps[:, 0] <= int(frame_idx)) & (int(frame_idx) <= cps[:, 1]))[0]
        if hits.size:
            out[idx] = float(shot_scores[int(hits[0])])
    return out


def select_by_knapsack(shot_scores: np.ndarray, nfps: np.ndarray, n_frames: int, summary_budget: float) -> np.ndarray:
    values = np.round(normalize_01(shot_scores) * 1000.0).astype(np.int32)
    capacity = int(int(n_frames) * float(summary_budget))
    selected_idx = [] if values.size == 0 or int(values.max()) <= 0 else vsumm_helper.knapsack(values.tolist(), nfps.astype(np.int32).reshape(-1).tolist(), capacity)
    selected = np.zeros((values.shape[0],), dtype=bool)
    selected[selected_idx] = True
    return selected


def summary_from_selected(selected: np.ndarray, cps: np.ndarray, n_frames: int) -> np.ndarray:
    summary = np.zeros((int(n_frames),), dtype=bool)
    for shot_idx, keep in enumerate(selected):
        if keep:
            first, last = cps[shot_idx]
            summary[int(first):int(last) + 1] = True
    return summary


def evaluate_scores(shot_scores, user_summary, gtscore, cps, n_frames, nfps, picks, oracle_selected, dataset, summary_budget, key):
    selected = select_by_knapsack(shot_scores, nfps, n_frames, summary_budget)
    pred_summary = summary_from_selected(selected, cps, n_frames)
    f1 = vsumm_helper.get_summ_f1score(pred_summ=pred_summary, test_summ=user_summary, eval_metric=infer_f1_metric_from_dataset(dataset))
    tau, rho = rank_or_nan(project_shot_scores_to_sampled(shot_scores, cps, picks), gtscore, key=key)
    precision, recall = precision_recall(selected, oracle_selected)
    return {
        'f1': float(f1), 'kendall': float(tau), 'spearman': float(rho),
        'oracle_recall': float(recall), 'oracle_precision': float(precision),
        'oracle_shot_f1': f1_binary(selected, oracle_selected),
        'selected_count': int(selected.sum()),
        'budget_ratio': float(pred_summary.sum() / max(1, int(n_frames))),
    }


def evaluate_video(key, group, utility_record, structured_entry, text_features, args):
    h5_key = Path(key).name
    gtscore = normalize_gtscore(group['gtscore'][...].astype(np.float32))
    user_summary = group['user_summary'][...].astype(np.float32)
    cps = group['change_points'][...].astype(np.int32)
    n_frames = scalar_int(group['n_frames'][...])
    nfps = group['n_frame_per_seg'][...].astype(np.int32).reshape(-1)
    picks = group['picks'][...].astype(np.int32).reshape(-1)
    oracle = exact_oracle_max_user(user_summary=user_summary, cps=cps, nfps=nfps, n_frames=n_frames, capacity=int(n_frames * float(args.summary_budget)))
    oracle_summary = build_summary_from_bitset(int(oracle['mask']), cps, n_frames)
    oracle_selected = bool_from_keyshot_summary(oracle_summary, cps)
    comp = build_components(utility_record)
    phase1_scores = normalize_01(comp['phase1_default'])
    closure = build_closure_components(utility_record, structured_entry, text_features, cps, n_frames, args)
    candidate_scores = closure['shot_scores']
    base = evaluate_scores(phase1_scores, user_summary, gtscore, cps, n_frames, nfps, picks, oracle_selected, args.dataset, args.summary_budget, f'{key}:phase1')
    cand = evaluate_scores(candidate_scores, user_summary, gtscore, cps, n_frames, nfps, picks, oracle_selected, args.dataset, args.summary_budget, f'{key}:closure')
    record = {
        'key': key, 'h5_key': h5_key, 'num_shots': int(cps.shape[0]), 'n_frames': int(n_frames),
        'base_f1': base['f1'], 'base_kendall': base['kendall'], 'base_spearman': base['spearman'], 'base_oracle_recall': base['oracle_recall'],
        'candidate_f1': cand['f1'], 'candidate_kendall': cand['kendall'], 'candidate_spearman': cand['spearman'], 'candidate_oracle_recall': cand['oracle_recall'],
        'candidate_oracle_precision': cand['oracle_precision'], 'candidate_oracle_shot_f1': cand['oracle_shot_f1'], 'candidate_budget_ratio': cand['budget_ratio'],
        'mean_closure_support': float(closure['closure_support'].mean()), 'max_closure_support': float(closure['closure_support'].max()) if closure['closure_support'].size else 0.0,
        'mean_unsupported_peak': float(closure['unsupported_peak'].mean()), 'max_unsupported_peak': float(closure['unsupported_peak'].max()) if closure['unsupported_peak'].size else 0.0,
        'num_closure_positive': int((closure['closure_support'] > 0).sum()), 'num_unsupported_positive': int((closure['unsupported_peak'] > 0).sum()),
    }
    teacher_record = {
        'shot_scores': candidate_scores.astype(np.float32), 'closure_support': closure['closure_support'].astype(np.float32),
        'unsupported_peak': closure['unsupported_peak'].astype(np.float32), 'peak_score': closure['peak_score'].astype(np.float32),
        'closure_lexical': closure['closure_lexical'].astype(np.float32), 'semantic_base': closure['semantic_base'].astype(np.float32),
        'meta': {'teacher_type': 'closure_aware_value_v2', 'h5_key': h5_key, 'uses_human_training_labels': False, 'alpha': float(args.alpha), 'beta': float(args.beta)},
    }
    return record, teacher_record


def mean(records, key):
    return safe_nanmean([record[key] for record in records], default=float('nan'))


def summarize(records, args):
    summary = {
        'num_videos': int(len(records)),
        'base_f1': mean(records, 'base_f1'), 'base_kendall': mean(records, 'base_kendall'), 'base_spearman': mean(records, 'base_spearman'), 'base_oracle_recall': mean(records, 'base_oracle_recall'),
        'candidate_f1': mean(records, 'candidate_f1'), 'candidate_kendall': mean(records, 'candidate_kendall'), 'candidate_spearman': mean(records, 'candidate_spearman'), 'candidate_oracle_recall': mean(records, 'candidate_oracle_recall'),
        'candidate_oracle_precision': mean(records, 'candidate_oracle_precision'), 'candidate_oracle_shot_f1': mean(records, 'candidate_oracle_shot_f1'), 'candidate_budget_ratio': mean(records, 'candidate_budget_ratio'),
        'mean_closure_support': mean(records, 'mean_closure_support'), 'mean_unsupported_peak': mean(records, 'mean_unsupported_peak'),
        'gate_f1': float(args.gate_f1), 'gate_tau': float(args.gate_tau), 'gate_rho': float(args.gate_rho), 'gate_recall': float(args.gate_recall),
    }
    summary['train_ready'] = bool(summary['candidate_f1'] > args.gate_f1 and summary['candidate_kendall'] > args.gate_tau and summary['candidate_spearman'] > args.gate_rho and summary['candidate_oracle_recall'] > args.gate_recall)
    summary['beats_computed_base_all_four'] = bool(summary['candidate_f1'] > summary['base_f1'] and summary['candidate_kendall'] > summary['base_kendall'] and summary['candidate_spearman'] > summary['base_spearman'] and summary['candidate_oracle_recall'] > summary['base_oracle_recall'])
    return summary


def write_csv(path: Path, records: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def maybe_write_teacher(path: str, teacher_records: Dict[str, Dict], train_ready: bool, force_write: bool):
    if not path:
        return
    if not train_ready and not force_write:
        print(f'SKIP_TEACHER_WRITE gate_failed path={path}')
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, teacher_records)
    print(f'WROTE_TEACHER {out}')


def main():
    args = parse_args()
    splits = load_all_splits(args.splits, val_ratio=args.val_ratio, seed=args.seed)
    validate_splits(splits, expected_dataset=args.dataset)
    keys = collect_unique_keys(splits, args.selection_part)
    if args.max_videos is not None:
        keys = keys[:args.max_videos]
    if not keys:
        raise ValueError('No selected keys.')
    structured = load_structured_captions(Path(args.structured_caption_path))
    utilities = load_shot_utility(Path(args.shot_utility_path))
    text_store = TextFeatureStore(Path(args.text_feature_path))
    h5_cache: Dict[str, h5py.File] = {}
    records: List[Dict] = []
    teacher_records: Dict[str, Dict] = {}
    try:
        for key in keys:
            h5_key, group = open_group(key, h5_cache)
            if h5_key not in structured:
                raise KeyError(f'Missing {h5_key} in structured captions')
            if h5_key not in utilities:
                raise KeyError(f'Missing {h5_key} in shot utility')
            record, teacher_record = evaluate_video(key, group, utilities[h5_key], structured[h5_key], text_store.get(h5_key), args)
            records.append(record)
            teacher_records[h5_key] = teacher_record
    finally:
        for handle in h5_cache.values():
            handle.close()
        text_store.close()
    summary = summarize(records, args)
    output = {
        'meta': {
            'dataset': args.dataset, 'splits': args.splits, 'selection_part': args.selection_part, 'seed': int(args.seed), 'val_ratio': float(args.val_ratio),
            'summary_budget': float(args.summary_budget), 'structured_caption_path': args.structured_caption_path, 'text_feature_path': args.text_feature_path, 'shot_utility_path': args.shot_utility_path,
            'teacher_type': 'closure_aware_value_v2', 'formula': 'value_v2 = phase1_default + alpha * closure_support - beta * unsupported_peak', 'alpha': float(args.alpha), 'beta': float(args.beta),
            'peak_quantile': float(args.peak_quantile), 'lookahead': int(args.lookahead), 'min_continuity': float(args.min_continuity),
            'uses_human_labels_for_training': False, 'human_label_use': 'teacher-alone validation gate only',
        },
        'summary': summary,
        'records': records,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    write_csv(Path(args.output_csv), records)
    maybe_write_teacher(args.teacher_output, teacher_records, summary['train_ready'], args.force_write)
    print(json.dumps({'summary': summary, 'output_json': str(out_json), 'output_csv': args.output_csv}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()