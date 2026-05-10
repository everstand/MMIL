import torch
from torch import nn
from torch.nn import functional as F


class ShotContextEncoder(nn.Module):
    """Text-conditioned temporal shot-token encoder.

    Shape contract:
        cond_frame_repr: [T, D]
        overlaps: [S, T]
        shot_lengths: [S]
        shot_text_feat: [S, D]
        shot_time_feat: [S, 4]
        return select_scores, rank_scores: [S]
    """

    def __init__(self,
                 num_feature: int,
                 num_hidden: int,
                 num_head: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 select_residual_scale: float = 0.5):
        super().__init__()
        if num_feature <= 0:
            raise ValueError(f'Invalid num_feature={num_feature}')
        if num_hidden <= 0:
            raise ValueError(f'Invalid num_hidden={num_hidden}')
        if num_head <= 0 or num_feature % num_head != 0:
            raise ValueError(
                f'num_feature must be divisible by num_head, got {num_feature} and {num_head}'
            )
        if num_layers <= 0:
            raise ValueError(f'Invalid num_layers={num_layers}')
        if select_residual_scale < 0:
            raise ValueError(f'Invalid select_residual_scale={select_residual_scale}')

        self.num_feature = int(num_feature)
        self.select_residual_scale = float(select_residual_scale)

        self.token_proj = nn.Sequential(
            nn.LayerNorm(num_feature * 2 + 4),
            nn.Linear(num_feature * 2 + 4, num_feature),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(num_feature),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=num_feature,
            nhead=num_head,
            dim_feedforward=max(num_hidden * 4, num_feature * 2),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.context_norm = nn.LayerNorm(num_feature)
        self.rank_head = nn.Linear(num_feature, 1)
        self.delta_head = nn.Linear(num_feature, 1)

    def forward(self,
                cond_frame_repr: torch.Tensor,
                overlaps: torch.Tensor,
                shot_lengths: torch.Tensor,
                shot_text_feat: torch.Tensor,
                shot_time_feat: torch.Tensor):
        select_logit, rank_logit, _delta_select = self.forward_logits(
            cond_frame_repr=cond_frame_repr,
            overlaps=overlaps,
            shot_lengths=shot_lengths,
            shot_text_feat=shot_text_feat,
            shot_time_feat=shot_time_feat,
        )
        return torch.sigmoid(select_logit), torch.sigmoid(rank_logit)

    def forward_logits(self,
                       cond_frame_repr: torch.Tensor,
                       overlaps: torch.Tensor,
                       shot_lengths: torch.Tensor,
                       shot_text_feat: torch.Tensor,
                       shot_time_feat: torch.Tensor):
        self._validate_inputs(
            cond_frame_repr=cond_frame_repr,
            overlaps=overlaps,
            shot_lengths=shot_lengths,
            shot_text_feat=shot_text_feat,
            shot_time_feat=shot_time_feat,
        )
        shot_visual = torch.matmul(overlaps, cond_frame_repr) / shot_lengths.clamp_min(1.0).unsqueeze(1)
        shot_token = torch.cat([shot_visual, shot_text_feat, shot_time_feat], dim=1)
        shot_token = self.token_proj(shot_token).unsqueeze(0)
        ctx = self.context_encoder(shot_token).squeeze(0)
        ctx = self.context_norm(ctx)

        rank_logit = self.rank_head(ctx).squeeze(-1)
        delta_select = self.select_residual_scale * torch.tanh(self.delta_head(ctx).squeeze(-1))
        select_logit = rank_logit + delta_select
        return select_logit, rank_logit, delta_select

    def _validate_inputs(self,
                         cond_frame_repr: torch.Tensor,
                         overlaps: torch.Tensor,
                         shot_lengths: torch.Tensor,
                         shot_text_feat: torch.Tensor,
                         shot_time_feat: torch.Tensor) -> None:
        if cond_frame_repr.ndim != 2:
            raise ValueError(f'Expected cond_frame_repr shape [T, D], got {tuple(cond_frame_repr.shape)}')
        if overlaps.ndim != 2:
            raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
        if shot_lengths.ndim != 1:
            raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
        if shot_text_feat.ndim != 2:
            raise ValueError(f'Expected shot_text_feat shape [S, D], got {tuple(shot_text_feat.shape)}')
        if shot_time_feat.ndim != 2 or shot_time_feat.shape[1] != 4:
            raise ValueError(f'Expected shot_time_feat shape [S, 4], got {tuple(shot_time_feat.shape)}')
        num_shots, seq_len = overlaps.shape
        if cond_frame_repr.shape[0] != seq_len:
            raise ValueError(
                f'overlaps/cond_frame_repr time mismatch: {seq_len} vs {cond_frame_repr.shape[0]}'
            )
        if cond_frame_repr.shape[1] != self.num_feature:
            raise ValueError(
                f'cond_frame_repr feature dim mismatch: {cond_frame_repr.shape[1]} vs {self.num_feature}'
            )
        if shot_text_feat.shape != (num_shots, self.num_feature):
            raise ValueError(
                f'shot_text_feat shape mismatch: {tuple(shot_text_feat.shape)} vs {(num_shots, self.num_feature)}'
            )
        if shot_time_feat.shape[0] != num_shots:
            raise ValueError(f'shot_time_feat shot mismatch: {shot_time_feat.shape[0]} vs {num_shots}')
        if shot_lengths.shape[0] != num_shots:
            raise ValueError(f'shot_lengths shot mismatch: {shot_lengths.shape[0]} vs {num_shots}')
        for name, value in (
            ('cond_frame_repr', cond_frame_repr),
            ('overlaps', overlaps),
            ('shot_lengths', shot_lengths),
            ('shot_text_feat', shot_text_feat),
            ('shot_time_feat', shot_time_feat),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite tensor in ShotContextEncoder: {name}')


def build_context_shot_text_stats(caption_spans_idx: torch.Tensor,
                                  caption_valid_mask: torch.Tensor,
                                  all_text_features: torch.Tensor,
                                  overlaps: torch.Tensor,
                                  shot_lengths: torch.Tensor,
                                  eps: float = 1e-6):
    """Aggregate caption text features onto shot tokens."""
    if caption_spans_idx.ndim != 2 or caption_spans_idx.shape[1] != 2:
        raise ValueError(f'Expected caption_spans_idx shape [M, 2], got {tuple(caption_spans_idx.shape)}')
    if caption_valid_mask.ndim != 1:
        raise ValueError(f'Expected caption_valid_mask shape [M], got {tuple(caption_valid_mask.shape)}')
    if all_text_features.ndim != 2:
        raise ValueError(f'Expected all_text_features shape [M, D], got {tuple(all_text_features.shape)}')
    if overlaps.ndim != 2:
        raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
    if shot_lengths.ndim != 1 or shot_lengths.shape[0] != overlaps.shape[0]:
        raise ValueError('shot_lengths must be [S] and match overlaps.')
    if caption_spans_idx.shape[0] != caption_valid_mask.shape[0]:
        raise ValueError('caption_spans_idx/caption_valid_mask length mismatch.')
    if caption_spans_idx.shape[0] != all_text_features.shape[0]:
        raise ValueError('caption_spans_idx/all_text_features length mismatch.')

    num_shots, seq_len = overlaps.shape
    feat_dim = all_text_features.shape[1]
    shot_text_sum = overlaps.new_zeros((num_shots, feat_dim))
    shot_text_mass = overlaps.new_zeros(num_shots)

    for k in range(caption_spans_idx.shape[0]):
        if float(caption_valid_mask[k].item()) <= 0.5:
            continue
        start_idx = int(caption_spans_idx[k, 0].item())
        end_idx = int(caption_spans_idx[k, 1].item())
        if end_idx < start_idx:
            continue
        start_idx = max(0, min(start_idx, seq_len - 1))
        end_idx = max(0, min(end_idx, seq_len - 1))
        caption_mask = overlaps.new_zeros(seq_len)
        caption_mask[start_idx:end_idx + 1] = 1.0
        weights = torch.matmul(overlaps, caption_mask)
        shot_text_sum = shot_text_sum + weights.unsqueeze(1) * all_text_features[k].unsqueeze(0)
        shot_text_mass = shot_text_mass + weights

    valid_shots = shot_text_mass > eps
    shot_text_feat = overlaps.new_zeros((num_shots, feat_dim))
    if bool(valid_shots.any().item()):
        shot_text_avg = shot_text_sum[valid_shots] / shot_text_mass[valid_shots].unsqueeze(1).clamp_min(eps)
        shot_text_feat[valid_shots] = F.normalize(shot_text_avg, p=2, dim=1)
    shot_mass_density = shot_text_mass / shot_lengths.clamp_min(1.0)
    return shot_text_feat, shot_mass_density, valid_shots


def build_shot_time_features(cps: torch.Tensor,
                             nfps: torch.Tensor,
                             n_frames: int) -> torch.Tensor:
    """Build [relative_center, relative_duration, gap_to_prev, gap_to_next]."""
    if cps.ndim != 2 or cps.shape[1] != 2:
        raise ValueError(f'Expected cps shape [S, 2], got {tuple(cps.shape)}')
    if nfps.ndim != 1 or nfps.shape[0] != cps.shape[0]:
        raise ValueError(f'Expected nfps shape [S], got {tuple(nfps.shape)} for cps {tuple(cps.shape)}')
    n_frames_float = float(int(n_frames))
    if n_frames_float <= 0:
        raise ValueError(f'Invalid n_frames={n_frames}')

    cps_float = cps.to(dtype=torch.float32)
    first = cps_float[:, 0]
    last = cps_float[:, 1]
    centers = ((first + last) * 0.5) / n_frames_float
    durations = nfps.to(dtype=torch.float32) / n_frames_float

    gaps_prev = torch.zeros_like(centers)
    gaps_next = torch.zeros_like(centers)
    if cps.shape[0] > 1:
        gaps = torch.clamp(first[1:] - last[:-1] - 1.0, min=0.0) / n_frames_float
        gaps_prev[1:] = gaps
        gaps_next[:-1] = gaps
    return torch.stack([centers, durations, gaps_prev, gaps_next], dim=1).clamp(0.0, 1.0)
