import logging

import numpy as np
import torch
from torch.nn import functional as F

from anchor_free.dsnet_af_mil import DSNetAFMIL
from evaluate_mil import evaluate_mil
from helpers import data_helper, mil_data_helper

logger = logging.getLogger(__name__)


def train(args, split, save_path):
    if 'val_keys' not in split:
        raise ValueError('train_mil requires split["val_keys"]. Do not use test_keys for checkpoint selection.')

    train_keys = split['train_keys']
    val_keys = split['val_keys']
    test_keys = split['test_keys']

    train_dataset_name = infer_single_dataset_name(train_keys)
    val_dataset_name = infer_single_dataset_name(val_keys)
    test_dataset_name = infer_single_dataset_name(test_keys)

    if len({train_dataset_name, val_dataset_name, test_dataset_name}) != 1:
        raise ValueError(
            f'Mixed dataset split is not allowed: '
            f'train={train_dataset_name}, val={val_dataset_name}, test={test_dataset_name}'
        )

    train_set = mil_data_helper.VideoDatasetMIL(train_keys)
    val_set = mil_data_helper.VideoDatasetMIL(val_keys)
    test_set = mil_data_helper.VideoDatasetMIL(test_keys)

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
    test_loader = data_helper.DataLoader(test_set, shuffle=False)

    save_path = str(save_path)
    if save_path.endswith('.pth'):
        kendall_save_path = save_path[:-4] + '_max_kendall.pth'
        spearman_save_path = save_path[:-4] + '_max_spearman.pth'
    else:
        kendall_save_path = save_path + '_max_kendall.pth'
        spearman_save_path = save_path + '_max_spearman.pth'

    best_val_fscore = -1.0
    kendall_at_best_fscore = 0.0
    spearman_at_best_fscore = 0.0

    max_val_kendall = -1.0
    fscore_at_max_kendall = 0.0
    spearman_at_max_kendall = 0.0

    max_val_spearman = -1.0
    fscore_at_max_spearman = 0.0
    kendall_at_max_spearman = 0.0

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss', 'bag_loss', 'num_effective')

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

            stats.update(loss=float(loss.item()), bag_loss=float(bag_loss.item()), num_effective=1.0)

        val_metrics = evaluate_mil(model=model, val_loader=val_loader, device=args.device)
        val_fscore = float(val_metrics['fscore'])
        val_kendall = float(val_metrics['kendall'])
        val_spearman = float(val_metrics['spearman'])

        if val_kendall > max_val_kendall:
            max_val_kendall = val_kendall
            fscore_at_max_kendall = val_fscore
            spearman_at_max_kendall = val_spearman
            torch.save(model.state_dict(), kendall_save_path)

        if val_spearman > max_val_spearman:
            max_val_spearman = val_spearman
            fscore_at_max_spearman = val_fscore
            kendall_at_max_spearman = val_kendall
            torch.save(model.state_dict(), spearman_save_path)

        if val_fscore > best_val_fscore:
            best_val_fscore = val_fscore
            kendall_at_best_fscore = val_kendall
            spearman_at_best_fscore = val_spearman
            torch.save(model.state_dict(), save_path)

        logger.info(
            'Epoch %03d/%03d | loss=%.4f | val_F1=%.4f | val_Rho=%.4f | best_val_F1=%.4f',
            epoch + 1,
            args.max_epoch,
            stats.loss,
            val_fscore,
            val_spearman,
            best_val_fscore,
        )

        logger.debug(
            'Epoch %03d/%03d detail | bag_loss=%.4f | effective_samples=%.4f | '
            'max_val_Rho=%.4f | Rho@best_F1=%.4f | F1@max_Rho=%.4f',
            epoch + 1,
            args.max_epoch,
            stats.bag_loss,
            stats.num_effective,
            max_val_spearman,
            spearman_at_best_fscore,
            fscore_at_max_spearman,
        )

    test_at_best_fscore = evaluate_checkpoint(model, save_path, test_loader, args.device)
    test_at_max_kendall = evaluate_checkpoint(model, kendall_save_path, test_loader, args.device)
    test_at_max_spearman = evaluate_checkpoint(model, spearman_save_path, test_loader, args.device)

    return {
        'val_best_fscore': float(best_val_fscore),
        'val_kendall_at_best_fscore': float(kendall_at_best_fscore),
        'val_spearman_at_best_fscore': float(spearman_at_best_fscore),
        'val_max_kendall': float(max_val_kendall),
        'val_fscore_at_max_kendall': float(fscore_at_max_kendall),
        'val_spearman_at_max_kendall': float(spearman_at_max_kendall),
        'val_max_spearman': float(max_val_spearman),
        'val_fscore_at_max_spearman': float(fscore_at_max_spearman),
        'val_kendall_at_max_spearman': float(kendall_at_max_spearman),
        'test_fscore_at_best_fscore': float(test_at_best_fscore['fscore']),
        'test_kendall_at_best_fscore': float(test_at_best_fscore['kendall']),
        'test_spearman_at_best_fscore': float(test_at_best_fscore['spearman']),
        'test_fscore_at_max_kendall': float(test_at_max_kendall['fscore']),
        'test_kendall_at_max_kendall': float(test_at_max_kendall['kendall']),
        'test_spearman_at_max_kendall': float(test_at_max_kendall['spearman']),
        'test_fscore_at_max_spearman': float(test_at_max_spearman['fscore']),
        'test_kendall_at_max_spearman': float(test_at_max_spearman['kendall']),
        'test_spearman_at_max_spearman': float(test_at_max_spearman['spearman']),
    }


def evaluate_checkpoint(model, ckpt_path, test_loader, device: str):
    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    return evaluate_mil(model=model, val_loader=test_loader, device=device)

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

