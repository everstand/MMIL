import logging

import numpy as np
import torch
from torch.nn import functional as F

from anchor_free.dsnet_af_mil import DSNetAFMIL
from evaluate_mil import evaluate_mil
from helpers import data_helper, mil_data_helper

logger = logging.getLogger(__name__)


def train(args, split, save_path):
    logger.info(f'lambda_smooth={args.lambda_smooth}, lambda_seg={args.lambda_seg}')
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

    save_path = str(save_path)
    if save_path.endswith('.pth'):
        spearman_save_path = save_path[:-4] + '_max_spearman.pth'
    else:
        spearman_save_path = save_path + '_max_spearman.pth'

    best_val_fscore = -1.0
    spearman_at_best_fscore = 0.0

    max_val_spearman = -1.0
    fscore_at_max_spearman = 0.0

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter(
            'loss',
            'bag_loss',
            ## 消融1:平滑损失 ,
            'smooth_loss',
            ## end 
            ## 消融2:segment consistency 
            'weighted_smooth_loss',
            'seg_loss',
            'weighted_seg_loss',
            ## end
            'num_effective',
        )

        for key, seq, soft_label, gtscore, user_summary, cps, n_frames, nfps, picks in train_loader:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32).to(args.device)

            normalized_target, is_effective = normalize_soft_label(soft_label_tensor)

            if not is_effective:
                stats.update(
                  loss=0.0, 
                  bag_loss=0.0, 
                  # 消融1:平滑损失
                  smooth_loss=0.0, 
                  # end
                  # 消融2:segment consistency
                  weighted_smooth_loss=0.0,
                  seg_loss=0.0,
                  weighted_seg_loss=0.0,
                  # end
                  num_effective=0.0)
                continue

            _, _, attn_weights, bag_logits = model(seq_tensor)

            bag_scores = torch.sigmoid(bag_logits)
            bag_loss = F.smooth_l1_loss(bag_scores, normalized_target)
            # loss = bag_loss
            # 消融1:平滑损失
            smooth_loss = compute_attention_smoothness_loss(attn_weights)
            # end
            # 消融2:segment consistency
            weighted_smooth_loss = args.lambda_smooth * smooth_loss

            seg_loss = compute_segment_consistency_loss(
                attn_weights=attn_weights,
                cps=cps,
                picks=picks,
                n_frames=n_frames,
            )
            weighted_seg_loss = args.lambda_seg * seg_loss
            # end
            loss = bag_loss + weighted_smooth_loss + weighted_seg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.update(
                loss=float(loss.item()),
                bag_loss=float(bag_loss.item()),
                # 消融1:平滑损失
                smooth_loss=float(smooth_loss.item()),
                # end
                # 消融2:segment consistency
                weighted_smooth_loss=float(weighted_smooth_loss.item()),
                seg_loss=float(seg_loss.item()),
                weighted_seg_loss=float(weighted_seg_loss.item()),
                # end
                num_effective=1.0,
            )

        val_fscore, val_spearman = evaluate_mil(
            model=model,
            val_loader=val_loader,
            device=args.device,
        )

        if val_spearman > max_val_spearman:
            max_val_spearman = val_spearman
            fscore_at_max_spearman = val_fscore
            torch.save(model.state_dict(), spearman_save_path)

        if val_fscore > best_val_fscore:
            best_val_fscore = val_fscore
            spearman_at_best_fscore = val_spearman
            torch.save(model.state_dict(), save_path)

        logger.info(
            f'Epoch: {epoch}/{args.max_epoch} '
            f'Loss: {stats.loss:.4f} '
            f'BagLoss: {stats.bag_loss:.4f} '
            # 消融1:平滑损失
            f'SmoothLoss: {stats.smooth_loss:.4f} '
            # end
            # 消融2:segment consistency
            f'WeightedSmoothLoss: {stats.weighted_smooth_loss:.4f} '
            f'SegLoss: {stats.seg_loss:.8e} '
            f'WeightedSegLoss: {stats.weighted_seg_loss:.8e} '
            # end
            f'EffectiveSamples: {stats.num_effective:.4f} '
            f'Val F-score cur/best: {val_fscore:.4f}/{best_val_fscore:.4f} '
            f'Val Spearman cur/max: {val_spearman:.4f}/{max_val_spearman:.4f} '
            f'Spearman@BestF: {spearman_at_best_fscore:.4f} '
            f'Fscore@MaxS: {fscore_at_max_spearman:.4f}'
        )

    return (
        best_val_fscore,
        spearman_at_best_fscore,
        max_val_spearman,
        fscore_at_max_spearman,
    )


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

 # 消融1:平滑损失 
def compute_attention_smoothness_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    if attn_weights.ndim != 1:
        raise ValueError(
            f'Expected attn_weights with shape [T], got {tuple(attn_weights.shape)}'
        )

    if attn_weights.shape[0] < 2:
        return attn_weights.new_zeros(())

    diff = attn_weights[1:] - attn_weights[:-1]
    return diff.abs().mean()
    # end

# 消融2:segment consistency
def compute_segment_consistency_loss(attn_weights: torch.Tensor,
                                     cps,
                                     picks,
                                     n_frames) -> torch.Tensor:
    if attn_weights.ndim != 1:
        raise ValueError(
            f'Expected attn_weights with shape [T], got {tuple(attn_weights.shape)}'
        )

    picks_np = np.asarray(picks, dtype=np.int64).reshape(-1)
    cps_np = np.asarray(cps, dtype=np.int64)
    n_frames_int = int(np.asarray(n_frames).item())

    if cps_np.ndim != 2 or cps_np.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {cps_np.shape}')

    if picks_np.shape[0] != attn_weights.shape[0]:
        raise ValueError(
            f'Length mismatch: len(picks)={picks_np.shape[0]} '
            f'vs len(attn_weights)={attn_weights.shape[0]}'
        )

    if n_frames_int <= 0:
        raise ValueError(f'Invalid n_frames: {n_frames_int}')

    interval_lo = picks_np
    interval_hi = np.empty_like(interval_lo)
    interval_hi[:-1] = picks_np[1:] - 1
    interval_hi[-1] = n_frames_int - 1

    seg_lo = cps_np[:, 0][:, None]
    seg_hi = cps_np[:, 1][:, None]
    int_lo = interval_lo[None, :]
    int_hi = interval_hi[None, :]

    overlap = np.minimum(seg_hi, int_hi) - np.maximum(seg_lo, int_lo) + 1
    overlap = np.clip(overlap, a_min=0, a_max=None).astype(np.float32)

    overlap_t = torch.tensor(
        overlap,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )

    seg_mass = overlap_t.sum(dim=1)
    valid = seg_mass > 0

    if not torch.any(valid):
        return attn_weights.new_zeros(())

    seg_mean = (overlap_t @ attn_weights) / seg_mass.clamp_min(1e-8)
    deviation = attn_weights.unsqueeze(0) - seg_mean.unsqueeze(1)
    seg_var = (overlap_t * deviation.pow(2)).sum(dim=1) / seg_mass.clamp_min(1e-8)

    return seg_var[valid].mean()