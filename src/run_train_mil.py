import argparse
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from anchor_free import train_mil as mil_trainer
from helpers.init_helper import init_logger, set_random_seed

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True,
                        choices=('tvsum', 'summe'))
    parser.add_argument('--splits', type=str, nargs='+', required=True)

    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--log-file', type=str, default='log_mil.txt')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.0)

    # 消融1:平滑损失
    parser.add_argument('--lambda-smooth', type=float, default=1e-3)
     # end 

    parser.add_argument('--base-model', type=str, default='attention',
                        choices=['attention', 'lstm', 'linear', 'bilstm', 'gcn'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)

    return parser


def main() -> None:
    args = get_parser().parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    init_logger(str(model_dir), args.log_file)

    logger.info('MIL training start')
    logger.info('Arguments: %s', vars(args))

    splits = load_all_splits(args.splits)
    validate_splits(splits, args.dataset)

    fscore_list: List[float] = []
    spearman_at_best_fscore_list: List[float] = []
    max_spearman_list: List[float] = []
    fscore_at_max_spearman_list: List[float] = []

    for split_idx, split in enumerate(splits):
        save_path = model_dir / f'best_model_split{split_idx}.pth'
        logger.info('Running split %d / %d', split_idx + 1, len(splits))

        best_fscore, spearman_at_best_fscore, max_spearman, fscore_at_max_spearman = mil_trainer.train(
            args=args,
            split=split,
            save_path=save_path,
        )
        fscore_list.append(float(best_fscore))
        spearman_at_best_fscore_list.append(float(spearman_at_best_fscore))
        max_spearman_list.append(float(max_spearman))

        logger.info(
            'Finished split %d / %d | best_fscore=%.4f | spearman_at_best_fscore=%.4f | max_spearman=%.4f | fscore_at_max_spearman=%.4f | checkpoint=%s',
            split_idx + 1,
            len(splits),
            best_fscore,
            spearman_at_best_fscore,
            max_spearman,
            fscore_at_max_spearman,
            str(save_path),
        )
        fscore_at_max_spearman_list.append(float(fscore_at_max_spearman))

    mean_fscore = sum(fscore_list) / len(fscore_list)
    mean_spearman_at_best_fscore = (
        sum(spearman_at_best_fscore_list) / len(spearman_at_best_fscore_list)
    )
    mean_max_spearman = sum(max_spearman_list) / len(max_spearman_list)
    mean_fscore_at_max_spearman = (
        sum(fscore_at_max_spearman_list) / len(fscore_at_max_spearman_list)
    )

    logger.info(
        'All splits finished | mean_fscore=%.4f | mean_spearman_at_best_fscore=%.4f | mean_max_spearman=%.4f | mean_fscore_at_max_spearman=%.4f',
        mean_fscore,
        mean_spearman_at_best_fscore,
        mean_max_spearman,
        mean_fscore_at_max_spearman,
    )


def load_all_splits(split_paths: List[str]) -> List[Dict]:
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

            normalized_split = {
                'train_keys': [normalize_split_key(key, split_root) for key in split['train_keys']],
                'test_keys': [normalize_split_key(key, split_root) for key in split['test_keys']],
            }
            all_splits.append(normalized_split)

    if not all_splits:
        raise ValueError('No splits loaded.')

    return all_splits


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
        test_keys = split['test_keys']

        if not train_keys:
            raise ValueError(f'Empty train_keys in split {split_idx}')
        if not test_keys:
            raise ValueError(f'Empty test_keys in split {split_idx}')

        for key in train_keys + test_keys:
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