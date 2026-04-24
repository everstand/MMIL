import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List

import yaml

from anchor_free import train_mil as mil_trainer
from helpers.init_helper import init_logger, set_random_seed

logger = logging.getLogger(__name__)

def mean_std(values: List[float]):
    values = [float(v) for v in values]
    if not values:
        return 0.0, 0.0

    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, choices=('tvsum', 'summe'))
    parser.add_argument('--splits', type=str, nargs='+', required=True)

    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--log-file', type=str, default='log_mil.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)

    parser.add_argument('--base-model', type=str, default='attention', choices=['attention'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)

    parser.add_argument('--val-ratio', type=float, default=0.2)

    return parser


def main() -> None:
    args = get_parser().parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    init_logger(str(model_dir), args.log_file)

    splits = load_all_splits(args.splits, val_ratio=args.val_ratio, seed=args.seed)
    validate_splits(splits, args.dataset)

    logger.info(
        'Run | dataset=%s | folds=%d | seed=%d | epochs=%d | lr=%.2e | wd=%.2e',
        args.dataset,
        len(splits),
        args.seed,
        args.max_epoch,
        args.lr,
        args.weight_decay,
    )
    logger.debug('Arguments: %s', vars(args))

    split_metrics: List[Dict[str, float]] = []

    for split_idx, split in enumerate(splits):
        save_path = model_dir / f'best_model_split{split_idx}.pth'

        metrics = mil_trainer.train(args=args, split=split, save_path=save_path)
        split_metrics.append(metrics)

        logger.info(
            'Split %d/%d | val_F1=%.4f | test_F1=%.4f | test_Tau=%.4f | test_Rho=%.4f',
            split_idx + 1,
            len(splits),
            metrics['val_best_fscore'],
            metrics['test_fscore_at_best_fscore'],
            metrics['test_kendall_at_best_fscore'],
            metrics['test_spearman_at_best_fscore'],
        )
        logger.debug(
            'Split %d/%d checkpoint=%s',
            split_idx + 1,
            len(splits),
            str(save_path),
        )

    test_fscore_list = [
        float(m['test_fscore_at_best_fscore'])
        for m in split_metrics
    ]
    test_kendall_list = [
        float(m['test_kendall_at_best_fscore'])
        for m in split_metrics
    ]
    test_spearman_list = [
        float(m['test_spearman_at_best_fscore'])
        for m in split_metrics
    ]

    mean_f1, std_f1 = mean_std(test_fscore_list)
    mean_tau, std_tau = mean_std(test_kendall_list)
    mean_rho, std_rho = mean_std(test_spearman_list)

    logger.info(
        'Final | test_F1=%.4f±%.4f | test_Tau=%.4f±%.4f | test_Rho=%.4f±%.4f',
        mean_f1,
        std_f1,
        mean_tau,
        std_tau,
        mean_rho,
        std_rho,
    )

def load_all_splits(split_paths: List[str], val_ratio: float, seed: int) -> List[Dict]:
    all_splits: List[Dict] = []

    for split_path in split_paths:
        path = Path(split_path)
        if not path.exists():
            raise FileNotFoundError(f'Split file not found: {path}')

        with open(path, 'r', encoding='utf-8') as f:
            obj = yaml.safe_load(f)

        if not isinstance(obj, list):
            raise ValueError(f'Split file must contain a list of folds: {path}')

        split_root = path.parent.resolve()

        for idx, split in enumerate(obj):
            if not isinstance(split, dict):
                raise ValueError(f'Invalid split entry at {path}, index {idx}')
            if 'train_keys' not in split or 'test_keys' not in split:
                raise ValueError(
                    f'Split entry must contain train_keys/test_keys: {path}, index {idx}'
                )

            train_keys = [normalize_split_key(key, split_root) for key in split['train_keys']]
            test_keys = [normalize_split_key(key, split_root) for key in split['test_keys']]

            if 'val_keys' in split and split['val_keys']:
                val_keys = [normalize_split_key(key, split_root) for key in split['val_keys']]
                train_keys_final = train_keys
            else:
                train_keys_final, val_keys = split_train_val(train_keys, val_ratio, seed + idx)

            all_splits.append({
                'train_keys': train_keys_final,
                'val_keys': val_keys,
                'test_keys': test_keys,
            })

    if not all_splits:
        raise ValueError('No splits loaded.')

    return all_splits


def split_train_val(train_keys: List[str], val_ratio: float, seed: int):
    if not train_keys:
        raise ValueError('Cannot split empty train_keys.')
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f'Invalid val_ratio={val_ratio}; expected 0 < val_ratio < 1.')

    keys = list(train_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)

    val_count = max(1, int(round(len(keys) * val_ratio)))
    if val_count >= len(keys):
        raise ValueError(
            f'val_ratio={val_ratio} leaves no training samples: train_count={len(keys)}, val_count={val_count}'
        )

    val_keys = sorted(keys[:val_count])
    train_keys_final = sorted(keys[val_count:])
    return train_keys_final, val_keys


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


def validate_splits(splits: List[Dict], expected_dataset: str) -> None:
    for split_idx, split in enumerate(splits):
        train_keys = split['train_keys']
        val_keys = split['val_keys']
        test_keys = split['test_keys']

        if not train_keys:
            raise ValueError(f'Empty train_keys in split {split_idx}')
        if not val_keys:
            raise ValueError(f'Empty val_keys in split {split_idx}')
        if not test_keys:
            raise ValueError(f'Empty test_keys in split {split_idx}')

        overlap_train_val = set(train_keys) & set(val_keys)
        overlap_train_test = set(train_keys) & set(test_keys)
        overlap_val_test = set(val_keys) & set(test_keys)
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                f'Split {split_idx} has overlapping train/val/test keys: '
                f'train_val={len(overlap_train_val)}, train_test={len(overlap_train_test)}, val_test={len(overlap_val_test)}'
            )

        for key in train_keys + val_keys + test_keys:
            dataset_name = infer_dataset_name_from_key(key)
            if dataset_name != expected_dataset:
                raise ValueError(
                    f'Dataset mismatch in split {split_idx}: '
                    f'expected "{expected_dataset}", got "{dataset_name}" from key "{key}"'
                )


def infer_dataset_name_from_key(key: str) -> str:
    key_lower = str(key).lower()
    if 'tvsum' in key_lower:
        return 'tvsum'
    if 'summe' in key_lower:
        return 'summe'
    raise ValueError(f'Cannot infer dataset name from key: {key}')


if __name__ == '__main__':
    main()
