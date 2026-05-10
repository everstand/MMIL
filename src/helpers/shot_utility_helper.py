# -*- coding: utf-8 -*-
"""Shot-level pseudo utility helper for optional ranking / selection training.

This module never reads human labels and never constructs a teacher from
gtscore/user_summary. It only reads offline `shot_utility.npy` records generated
by `src/make_shot_pseudo_utility.py`.
"""

from pathlib import Path
from typing import Dict, Callable, Optional, Tuple

import numpy as np

from helpers import vsumm_helper
from helpers.mil_path_helper import get_dataset_pseudo_dir


def resolve_shot_utility_path(dataset_name: str,
                              explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return get_dataset_pseudo_dir(dataset_name) / 'shot_utility.npy'


def normalize_01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('normalize_01 received non-finite values.')

    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < eps:
        return np.zeros_like(values, dtype=np.float32)

    return ((values - lo) / (hi - lo + eps)).astype(np.float32)


def get_component(record: Dict, name: str) -> np.ndarray:
    if name not in record:
        raise KeyError(f'Missing shot-utility component "{name}".')
    arr = np.asarray(record[name], dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f'Empty shot-utility component "{name}".')
    if not np.isfinite(arr).all():
        raise ValueError(f'Non-finite values in shot-utility component "{name}".')
    return arr


def get_optional_component(record: Dict, name: str, length: int) -> Tuple[np.ndarray, bool]:
    if name not in record:
        return np.zeros((length,), dtype=np.float32), False
    arr = get_component(record, name)
    if arr.shape[0] != length:
        raise ValueError(
            f'Shot-utility optional component length mismatch for {name}: '
            f'{arr.shape[0]} vs expected {length}'
        )
    return arr, True


def build_components(record: Dict) -> Dict:
    semantic = get_component(record, 'semantic_coverage')
    representativeness = get_component(record, 'visual_representativeness')
    redundancy = get_component(record, 'redundancy_penalty')
    eventiveness = get_component(record, 'eventiveness')
    phase1_default = get_component(record, 'shot_utility')

    lengths = {
        semantic.shape[0],
        representativeness.shape[0],
        redundancy.shape[0],
        eventiveness.shape[0],
        phase1_default.shape[0],
    }
    if len(lengths) != 1:
        raise ValueError(
            'Shot-utility component length mismatch: '
            f'semantic={semantic.shape}, rep={representativeness.shape}, '
            f'redundancy={redundancy.shape}, eventiveness={eventiveness.shape}, '
            f'phase1_default={phase1_default.shape}'
        )

    length = int(phase1_default.shape[0])
    local_caption, has_local_caption = get_optional_component(
        record, 'local_caption_similarity_raw', length
    )
    global_caption, has_global_caption = get_optional_component(
        record, 'global_caption_similarity_raw', length
    )
    caption_change, has_caption_change = get_optional_component(
        record, 'caption_change_raw', length
    )
    visual_change, has_visual_change = get_optional_component(
        record, 'visual_change_raw', length
    )

    rep_n = normalize_01(representativeness)
    red_n = normalize_01(redundancy)
    local_caption_n = normalize_01(local_caption)
    global_caption_n = normalize_01(global_caption)
    caption_change_n = normalize_01(caption_change)
    visual_change_n = normalize_01(visual_change)
    caption_mgs = normalize_01(0.7 * local_caption_n + 0.3 * global_caption_n)

    return {
        'phase1_default': normalize_01(phase1_default),
        'semantic': normalize_01(semantic),
        'representativeness': rep_n,
        'distinctiveness': normalize_01(1.0 - rep_n),
        'redundancy': red_n,
        'anti_redundancy': normalize_01(1.0 - red_n),
        'eventiveness': normalize_01(eventiveness),
        'caption_local': local_caption_n,
        'caption_global': global_caption_n,
        'caption_mgs': caption_mgs,
        'caption_change': caption_change_n,
        'visual_change': visual_change_n,
        'caption_prior_available': has_local_caption and has_global_caption,
        'change_prior_available': has_caption_change or has_visual_change,
    }


FormulaFn = Callable[[Dict], np.ndarray]


def formula_definitions() -> Dict[str, FormulaFn]:
    def n(x):
        return normalize_01(x)

    def require(c: Dict, name: str, available_flag: str) -> np.ndarray:
        if not bool(c.get(available_flag, False)):
            raise KeyError(
                f'Formula requires unavailable shot-utility component group: '
                f'{available_flag}'
            )
        return c[name]

    return {
        'phase1_default': lambda c: c['phase1_default'],

        'semantic': lambda c: c['semantic'],
        'representativeness': lambda c: c['representativeness'],
        'distinctiveness': lambda c: c['distinctiveness'],
        'anti_redundancy': lambda c: c['anti_redundancy'],
        'eventiveness': lambda c: c['eventiveness'],

        # Caption-summary-prior candidates from multi-grained saliency scoring.
        'caption_mgs': lambda c: require(
            c, 'caption_mgs', 'caption_prior_available'
        ),
        'caption_mgs_plus_change': lambda c: n(
            require(c, 'caption_mgs', 'caption_prior_available')
            + 0.2 * require(c, 'caption_change', 'change_prior_available')
            + 0.1 * require(c, 'visual_change', 'change_prior_available')
        ),
        'caption_mgs_plus_event': lambda c: n(
            require(c, 'caption_mgs', 'caption_prior_available')
            + 0.25 * c['eventiveness']
        ),
        'caption_mgs_plus_distinct': lambda c: n(
            require(c, 'caption_mgs', 'caption_prior_available')
            + 0.25 * c['distinctiveness']
        ),
        'caption_mgs_plus_event_minus_red': lambda c: n(
            require(c, 'caption_mgs', 'caption_prior_available')
            + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']
        ),
        'caption_mgs_rank_safe': lambda c: n(
            require(c, 'caption_mgs', 'caption_prior_available')
            + 0.25 * c['distinctiveness']
            + 0.15 * c['eventiveness'] - 0.1 * c['redundancy']
        ),

        'semantic_plus_rep': lambda c: n(c['semantic'] + c['representativeness']),
        'semantic_plus_distinct': lambda c: n(c['semantic'] + c['distinctiveness']),
        'semantic_plus_anti_redundancy': lambda c: n(c['semantic'] + c['anti_redundancy']),
        'rep_plus_anti_redundancy': lambda c: n(c['representativeness'] + c['anti_redundancy']),
        'distinct_plus_anti_redundancy': lambda c: n(c['distinctiveness'] + c['anti_redundancy']),

        'semantic_minus_red': lambda c: n(c['semantic'] - 0.2 * c['redundancy']),
        'rep_minus_red': lambda c: n(c['representativeness'] - 0.2 * c['redundancy']),
        'distinct_minus_red': lambda c: n(c['distinctiveness'] - 0.2 * c['redundancy']),
        'semantic_plus_rep_minus_red': lambda c: n(
            c['semantic'] + 0.5 * c['representativeness'] - 0.2 * c['redundancy']
        ),
        'semantic_plus_distinct_minus_red': lambda c: n(
            c['semantic'] + 0.5 * c['distinctiveness'] - 0.2 * c['redundancy']
        ),

        'semantic_x_anti_redundancy': lambda c: n(c['semantic'] * c['anti_redundancy']),
        'rep_x_anti_redundancy': lambda c: n(c['representativeness'] * c['anti_redundancy']),
        'semantic_x_rep': lambda c: n(c['semantic'] * c['representativeness']),
        'semantic_x_distinct': lambda c: n(c['semantic'] * c['distinctiveness']),

        'semantic_plus_event': lambda c: n(c['semantic'] + 0.25 * c['eventiveness']),
        'semantic_plus_rep_plus_event': lambda c: n(
            c['semantic'] + 0.5 * c['representativeness'] + 0.25 * c['eventiveness']
        ),
        'semantic_plus_distinct_plus_event': lambda c: n(
            c['semantic'] + 0.5 * c['distinctiveness'] + 0.25 * c['eventiveness']
        ),
        'semantic_plus_event_minus_red': lambda c: n(
            c['semantic'] + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']
        ),
        'semantic_plus_rep_plus_event_minus_red': lambda c: n(
            c['semantic'] + 0.5 * c['representativeness']
            + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']
        ),
        'semantic_plus_distinct_plus_event_minus_red': lambda c: n(
            c['semantic'] + 0.5 * c['distinctiveness']
            + 0.25 * c['eventiveness'] - 0.2 * c['redundancy']
        ),
    }


def compute_formula_utility(record: Dict, formula_name: str) -> np.ndarray:
    formulas = formula_definitions()
    if formula_name not in formulas:
        raise KeyError(
            f'Unknown utility formula "{formula_name}". '
            f'Available formulas: {sorted(formulas.keys())}'
        )

    components = build_components(record)
    utility = normalize_01(formulas[formula_name](components))
    if not np.isfinite(utility).all():
        raise ValueError(f'Non-finite utility produced by formula "{formula_name}".')
    return utility.astype(np.float32)


def build_budgeted_pseudo_summary_masks(utility: np.ndarray,
                                        cps: np.ndarray,
                                        nfps: np.ndarray,
                                        n_frames: int,
                                        summary_budget: float,
                                        negative_quantile: float) -> Dict[str, np.ndarray]:
    utility = normalize_01(utility)
    cps = np.asarray(cps, dtype=np.int32)
    nfps = np.asarray(nfps, dtype=np.int32).reshape(-1)

    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {cps.shape}')
    if utility.shape[0] != cps.shape[0]:
        raise ValueError(f'utility/cps length mismatch: {utility.shape[0]} vs {cps.shape[0]}')
    if nfps.shape[0] != cps.shape[0]:
        raise ValueError(f'nfps/cps length mismatch: {nfps.shape[0]} vs {cps.shape[0]}')
    if n_frames <= 0:
        raise ValueError(f'Invalid n_frames: {n_frames}')
    if not (0.0 < summary_budget < 1.0):
        raise ValueError(f'Invalid summary_budget={summary_budget}; expected 0 < budget < 1.')
    if not (0.0 < negative_quantile < 1.0):
        raise ValueError(
            f'Invalid negative_quantile={negative_quantile}; expected 0 < q < 1.'
        )

    values = np.round(utility * 1000.0).astype(np.int32)
    capacity = int(n_frames * summary_budget)

    if values.size == 0 or int(values.max()) <= 0:
        selected_idx = []
    else:
        selected_idx = vsumm_helper.knapsack(values.tolist(), nfps.tolist(), capacity)

    selected = np.zeros(cps.shape[0], dtype=bool)
    selected[selected_idx] = True

    neg_thr = float(np.quantile(utility, negative_quantile))
    negative = (~selected) & (utility <= neg_thr)
    ignore = ~(selected | negative)

    target = np.zeros(cps.shape[0], dtype=np.float32)
    supervised = np.zeros(cps.shape[0], dtype=bool)

    target[selected] = 1.0
    target[negative] = 0.0
    supervised[selected | negative] = True

    return {
        'utility': utility.astype(np.float32),
        'selected_mask': selected,
        'negative_mask': negative,
        'ignore_mask': ignore,
        'target': target,
        'supervised_mask': supervised,
        'negative_threshold': np.asarray(neg_thr, dtype=np.float32),
    }


class ShotUtilityStore(object):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.records = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> Dict[str, Dict]:
        if not path.exists():
            raise FileNotFoundError(f'Shot utility file not found: {path}')

        obj = np.load(path, allow_pickle=True)
        try:
            obj = obj.item()
        except Exception as exc:
            raise ValueError(f'Invalid shot utility file format: {path}') from exc

        if not isinstance(obj, dict):
            raise ValueError(
                f'Shot utility file must contain dict[h5_key -> dict], got {type(obj)}'
            )

        if not obj:
            raise ValueError(f'Empty shot utility file: {path}')

        return obj

    def get(self, h5_key: str, formula_name: str) -> np.ndarray:
        if h5_key not in self.records:
            raise KeyError(f'Missing h5 key "{h5_key}" in shot utility file: {self.path}')
        record = self.records[h5_key]
        if not isinstance(record, dict):
            raise ValueError(f'Shot utility record for "{h5_key}" must be dict, got {type(record)}')
        return compute_formula_utility(record, formula_name=formula_name)

    def get_budgeted_masks(self,
                           h5_key: str,
                           formula_name: str,
                           cps: np.ndarray,
                           nfps: np.ndarray,
                           n_frames: int,
                           summary_budget: float,
                           negative_quantile: float) -> Dict[str, np.ndarray]:
        utility = self.get(h5_key=h5_key, formula_name=formula_name)
        return build_budgeted_pseudo_summary_masks(
            utility=utility,
            cps=cps,
            nfps=nfps,
            n_frames=n_frames,
            summary_budget=summary_budget,
            negative_quantile=negative_quantile,
        )
