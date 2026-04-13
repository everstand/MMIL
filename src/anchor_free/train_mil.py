import logging

import numpy as np
import torch
from torch.nn import functional as F

from anchor_free.dsnet_af_mil import DSNetAFMIL
from evaluate_mil import evaluate_mil
from helpers import data_helper, mil_data_helper

logger = logging.getLogger(__name__)


def train(args, split, save_path):
    train_keys = split['train_keys']
    val_keys = split['test_keys']

    train_dataset_name = infer_single_dataset_name(train_keys)
    val_dataset_name = infer_single_dataset_name(val_keys)

    if train_dataset_name != val_dataset_name:
        raise ValueError(
            f'Mixed dataset split is not allowed: train={train_dataset_name}, val={val_dataset_name}'
        )

    train_set = mil_data_helper.VideoDatasetMIL(train_keys)
    val_set = mil_data_helper.VideoDatasetMIL(val_keys)

    num_classes = infer_num_classes(train_set)
    model = DSNetAFMIL(
        base_model=args.base_model,
        num_feature=args.num_feature,
        num_hidden=args.num_hidden,
        num_head=args.num_head,
        num_classes=num_classes,
    ).to(args.device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_loader = data_helper.DataLoader(train_set, shuffle=True)
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    best_val_fscore = -1.0
    best_val_spearman = 0.0

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter(
            'loss',
            'bag_loss',
            'num_effective',
        )

        for key, seq, soft_label, gtscore, user_summary, cps, n_frames, nfps, picks in train_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32).to(args.device)

            normalized_target, is_effective = normalize_soft_label(soft_label_tensor)

            if not is_effective:
                stats.update(loss=0.0, bag_loss=0.0, num_effective=0.0)
                continue

            _, _, attn_weights, bag_logits = model(seq_tensor)

            bag_scores = torch.sigmoid(bag_logits)
            bag_loss = F.smooth_l1_loss(bag_scores, normalized_target)
            loss = bag_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(
                loss=float(loss.item()),
                bag_loss=float(bag_loss.item()),
                num_effective=1.0,
            )

        val_fscore, val_spearman = evaluate_mil(
            model=model,
            val_loader=val_loader,
            device=args.device,
        )

        if val_fscore > best_val_fscore:
            best_val_fscore = val_fscore
            best_val_spearman = val_spearman
            torch.save(model.state_dict(), str(save_path))

        logger.info(
            f'Epoch: {epoch}/{args.max_epoch} '
            f'Loss: {stats.loss:.4f} '
            f'BagLoss: {stats.bag_loss:.4f} '
            f'EffectiveSamples: {stats.num_effective:.4f} '
            f'Val F-score cur/best: {val_fscore:.4f}/{best_val_fscore:.4f} '
            f'Val Spearman cur/best: {val_spearman:.4f}/{best_val_spearman:.4f}'
        )

    return best_val_fscore


def infer_single_dataset_name(keys):
    names = {infer_dataset_name_from_key(key) for key in keys}
    if len(names) != 1:
        raise ValueError(f'Expected a single dataset in one run, got: {sorted(names)}')
    return next(iter(names))


def infer_dataset_name_from_key(key: str) -> str:
    key_lower = str(key).lower()
    if 'tvsum' in key_lower:
        return 'tvsum'
    if 'summe' in key_lower:
        return 'summe'
    raise ValueError(f'Cannot infer dataset name from key: {key}')


def infer_num_classes(dataset) -> int:
    if len(dataset) == 0:
        raise ValueError('Cannot infer num_classes from empty dataset.')
    sample = dataset[0]
    soft_label = np.asarray(sample[2], dtype=np.float32)
    if soft_label.ndim != 1:
        raise ValueError(f'Invalid soft_label shape: {soft_label.shape}')
    return int(soft_label.shape[0])


def normalize_soft_label(soft_label: torch.Tensor, eps: float = 1e-8):
    label_min = torch.min(soft_label)
    label_max = torch.max(soft_label)
    label_range = label_max - label_min

    if float(label_range.item()) < eps:
        normalized = torch.zeros_like(soft_label)
        return normalized, False

    normalized = (soft_label - label_min) / (label_range + eps)
    return normalized, True