from torch import nn
import torch

from modules.models import build_base_model
from modules.shot_context_encoder import ShotContextEncoder


def minmax_normalize_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = torch.min(x)
    x_max = torch.max(x)
    if float((x_max - x_min).item()) <= eps:
        return torch.ones_like(x) * 0.5
    return (x - x_min) / (x_max - x_min + eps)


class DSNetAFMILCond(nn.Module):
    """Conditioned MIL DSNet.

    score_head='single':
        Backward-compatible path. The softmax pooling weights are also used as
        summary scores.

    score_head='dual':
        Decouples MIL pooling from summary selection:
            pool_weights      -> bag_logits / summary_feat
            selection_scores  -> evaluation / selection losses

    score_head='residual_dual':
        Uses the normalized pooling score as a frozen anchor and learns a
        residual selection correction. This is intended for small/unstable
        datasets where a randomly initialized selection head may destroy a
        stable single-head baseline.
    """

    def __init__(self,
                 base_model: str,
                 num_feature: int,
                 num_hidden: int,
                 num_head: int,
                 num_classes: int,
                 score_head: str = 'single',
                 use_shot_head: bool = False,
                 shot_head_mode: str = 'single'):
        super().__init__()

        if score_head not in ('single', 'dual', 'residual_dual'):
            raise ValueError(
                f'Invalid score_head={score_head}; expected single, dual, or residual_dual.'
            )
        if shot_head_mode not in ('single', 'dual', 'context'):
            raise ValueError(
                f'Invalid shot_head_mode={shot_head_mode}; expected single, dual, or context.'
            )

        self.num_classes = num_classes
        self.num_feature = num_feature
        self.score_head = score_head
        self.use_shot_head = bool(use_shot_head)
        self.shot_head_mode = shot_head_mode

        self.base_model = build_base_model(base_model, num_feature, num_head)

        self.layer_norm = nn.LayerNorm(num_feature)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=num_feature,
            num_heads=num_head,
            batch_first=True,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden),
        )

        self.fc_cls = nn.Linear(num_hidden, num_classes)
        self.fc_attn = nn.Linear(num_hidden, 1)

        if self.score_head in ('dual', 'residual_dual'):
            self.fc_select = nn.Linear(num_hidden, 1)
            if self.score_head == 'residual_dual':
                nn.init.zeros_(self.fc_select.weight)
                nn.init.zeros_(self.fc_select.bias)
        else:
            self.fc_select = None

        self.shot_context_encoder = None
        if self.use_shot_head:
            if self.shot_head_mode == 'context':
                self.shot_score_head = None
                self.shot_rank_head = None
                self.shot_context_encoder = ShotContextEncoder(
                    num_feature=num_feature,
                    num_hidden=num_hidden,
                    num_head=num_head,
                )
            else:
                self.shot_score_head = nn.Sequential(
                    nn.LayerNorm(num_feature),
                    nn.Linear(num_feature, num_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(num_hidden, 1),
                )
                if self.shot_head_mode == 'dual':
                    self.shot_rank_head = nn.Sequential(
                        nn.LayerNorm(num_feature),
                        nn.Linear(num_feature, num_hidden),
                        nn.ReLU(inplace=True),
                        nn.Linear(num_hidden, 1),
                    )
                else:
                    self.shot_rank_head = None
        else:
            self.shot_score_head = None
            self.shot_rank_head = None

    def forward(self,
                x: torch.Tensor,
                text_cond: torch.Tensor,
                text_cond_mask: torch.Tensor = None):
        if x.ndim != 3:
            raise ValueError(f'Expected x shape [B, T, D], got {tuple(x.shape)}')
        if x.shape[0] != 1:
            raise ValueError(
                f'DSNetAFMILCond expects batch size 1 in current training pipeline, got {x.shape[0]}'
            )

        if text_cond.ndim == 2:
            text_cond = text_cond.unsqueeze(0)
        elif text_cond.ndim != 3:
            raise ValueError(
                f'Expected text_cond shape [M, D] or [B, M, D], got {tuple(text_cond.shape)}'
            )

        if text_cond.shape[0] != 1:
            raise ValueError(
                f'DSNetAFMILCond expects text_cond batch size 1, got {text_cond.shape[0]}'
            )
        if x.shape[2] != self.num_feature:
            raise ValueError(
                f'Input feature dim mismatch: got {x.shape[2]}, expected {self.num_feature}'
            )
        if text_cond.shape[2] != self.num_feature:
            raise ValueError(
                f'Text feature dim mismatch: got {text_cond.shape[2]}, expected {self.num_feature}'
            )

        key_padding_mask = self.build_text_key_padding_mask(
            text_cond=text_cond,
            text_cond_mask=text_cond_mask,
        )

        raw_x = x

        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)

        pre_cross_frame_repr = out.squeeze(0)

        cond_out, _ = self.cross_attn(
            query=out,
            key=text_cond,
            value=text_cond,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        cond_out = self.cross_attn_layer_norm(cond_out + out)
        cond_frame_repr = cond_out.squeeze(0)

        hidden = self.fc1(cond_out).squeeze(0)
        hidden_frame_repr = hidden
        raw_frame_features = raw_x.squeeze(0)

        instance_logits = self.fc_cls(hidden)

        pool_logits = self.fc_attn(hidden).squeeze(-1)
        pool_weights = torch.softmax(pool_logits, dim=0)

        if self.score_head == 'dual':
            selection_logits = self.fc_select(hidden).squeeze(-1)
            summary_scores = torch.sigmoid(selection_logits)
        elif self.score_head == 'residual_dual':
            residual_logits = self.fc_select(hidden).squeeze(-1)
            anchor_scores = pool_weights.detach().clamp(1e-6, 1.0 - 1e-6)
            anchor_logits = torch.logit(anchor_scores)
            summary_scores = torch.sigmoid(anchor_logits + residual_logits)
        else:
            summary_scores = pool_weights

        bag_logits = torch.sum(
            pool_weights.unsqueeze(-1) * instance_logits,
            dim=0,
        )

        summary_feat = torch.sum(
            pool_weights.unsqueeze(-1) * raw_frame_features,
            dim=0,
        )

        return (
            instance_logits,
            pool_logits,
            summary_scores,
            bag_logits,
            summary_feat,
            pre_cross_frame_repr,
            cond_frame_repr,
            hidden_frame_repr,
        )

    @torch.no_grad()
    def build_text_key_padding_mask(self,
                                    text_cond: torch.Tensor,
                                    text_cond_mask: torch.Tensor = None):
        if text_cond_mask is None:
            return None

        if text_cond_mask.ndim == 1:
            text_cond_mask = text_cond_mask.unsqueeze(0)
        elif text_cond_mask.ndim != 2:
            raise ValueError(
                f'Expected text_cond_mask shape [M] or [B, M], got {tuple(text_cond_mask.shape)}'
            )

        if text_cond_mask.shape[0] != text_cond.shape[0]:
            raise ValueError(
                f'text_cond/text_cond_mask batch mismatch: '
                f'{text_cond.shape[0]} vs {text_cond_mask.shape[0]}'
            )
        if text_cond_mask.shape[1] != text_cond.shape[1]:
            raise ValueError(
                f'text_cond/text_cond_mask length mismatch: '
                f'{text_cond.shape[1]} vs {text_cond_mask.shape[1]}'
            )

        valid_mask = text_cond_mask.to(device=text_cond.device) > 0.5
        if not bool(valid_mask.any().item()):
            raise ValueError('text_cond_mask masks out all text condition tokens.')
        return ~valid_mask

    @torch.no_grad()
    def predict_summary_scores(self,
                               seq: torch.Tensor,
                               text_cond: torch.Tensor,
                               text_cond_mask: torch.Tensor = None) -> torch.Tensor:
        _, _, summary_scores, _, _, _, _, _ = self(seq, text_cond, text_cond_mask)
        return summary_scores

    def predict_shot_scores(self,
                            frame_repr: torch.Tensor,
                            overlaps: torch.Tensor,
                            shot_lengths: torch.Tensor,
                            head: str = 'selection',
                            shot_text_feat: torch.Tensor = None,
                            shot_time_feat: torch.Tensor = None) -> torch.Tensor:
        """Predict direct shot scores from frame representations.

        By default callers should pass the text-conditioned frame representation
        returned by `forward` as `cond_frame_repr`, not the pre-cross-attention
        representation.

        Shape contract:
            frame_repr: [T, D]
            overlaps: [S, T]
            shot_lengths: [S]
            return: [S]
        """
        if not self.use_shot_head:
            raise RuntimeError('Shot score head is disabled for this model.')
        if head not in ('selection', 'rank'):
            raise ValueError(f'Invalid shot score head={head}; expected selection or rank.')
        if frame_repr.ndim != 2:
            raise ValueError(f'Expected frame_repr shape [T, D], got {tuple(frame_repr.shape)}')
        if overlaps.ndim != 2:
            raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
        if shot_lengths.ndim != 1:
            raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
        if frame_repr.shape[1] != self.num_feature:
            raise ValueError(
                f'frame_repr feature dim mismatch: got {frame_repr.shape[1]}, expected {self.num_feature}'
            )
        if overlaps.shape[1] != frame_repr.shape[0]:
            raise ValueError(
                f'overlaps/frame_repr time mismatch: {overlaps.shape[1]} vs {frame_repr.shape[0]}'
            )
        if overlaps.shape[0] != shot_lengths.shape[0]:
            raise ValueError(
                f'overlaps/shot_lengths shot mismatch: {overlaps.shape[0]} vs {shot_lengths.shape[0]}'
            )

        if self.shot_head_mode == 'context':
            selection_scores, rank_scores = self.predict_shot_score_pair(
                frame_repr=frame_repr,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
                shot_text_feat=shot_text_feat,
                shot_time_feat=shot_time_feat,
            )
            return rank_scores if head == 'rank' else selection_scores

        shot_repr = torch.matmul(overlaps, frame_repr) / shot_lengths.clamp_min(1.0).unsqueeze(1)
        if head == 'rank':
            if self.shot_rank_head is None:
                raise RuntimeError('Shot rank head is disabled; use shot_head_mode=dual or context.')
            shot_logits = self.shot_rank_head(shot_repr).squeeze(-1)
        else:
            shot_logits = self.shot_score_head(shot_repr).squeeze(-1)
        return torch.sigmoid(shot_logits)

    def predict_shot_logit_pair(self,
                                frame_repr: torch.Tensor,
                                overlaps: torch.Tensor,
                                shot_lengths: torch.Tensor,
                                shot_text_feat: torch.Tensor = None,
                                shot_time_feat: torch.Tensor = None):
        if not self.use_shot_head:
            raise RuntimeError('Shot score head is disabled for this model.')
        if frame_repr.ndim != 2:
            raise ValueError(f'Expected frame_repr shape [T, D], got {tuple(frame_repr.shape)}')
        if overlaps.ndim != 2:
            raise ValueError(f'Expected overlaps shape [S, T], got {tuple(overlaps.shape)}')
        if shot_lengths.ndim != 1:
            raise ValueError(f'Expected shot_lengths shape [S], got {tuple(shot_lengths.shape)}')
        if overlaps.shape[1] != frame_repr.shape[0]:
            raise ValueError(
                f'overlaps/frame_repr time mismatch: {overlaps.shape[1]} vs {frame_repr.shape[0]}'
            )
        if overlaps.shape[0] != shot_lengths.shape[0]:
            raise ValueError(
                f'overlaps/shot_lengths shot mismatch: {overlaps.shape[0]} vs {shot_lengths.shape[0]}'
            )
        if self.shot_head_mode == 'context':
            if self.shot_context_encoder is None:
                raise RuntimeError('Shot context encoder is disabled.')
            if shot_text_feat is None or shot_time_feat is None:
                raise ValueError('shot_head_mode=context requires shot_text_feat and shot_time_feat.')
            return self.shot_context_encoder.forward_logits(
                cond_frame_repr=frame_repr,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
                shot_text_feat=shot_text_feat,
                shot_time_feat=shot_time_feat,
            )

        shot_repr = torch.matmul(overlaps, frame_repr) / shot_lengths.clamp_min(1.0).unsqueeze(1)
        selection_logit = self.shot_score_head(shot_repr).squeeze(-1)
        if self.shot_head_mode == 'dual':
            rank_logit = self.shot_rank_head(shot_repr).squeeze(-1)
            delta_select = selection_logit - rank_logit
        else:
            rank_logit = selection_logit
            delta_select = torch.zeros_like(selection_logit)
        return selection_logit, rank_logit, delta_select

    def predict_shot_score_pair(self,
                                frame_repr: torch.Tensor,
                                overlaps: torch.Tensor,
                                shot_lengths: torch.Tensor,
                                shot_text_feat: torch.Tensor = None,
                                shot_time_feat: torch.Tensor = None):
        if self.shot_head_mode == 'context':
            select_logit, rank_logit, _delta_select = self.predict_shot_logit_pair(
                frame_repr=frame_repr,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
                shot_text_feat=shot_text_feat,
                shot_time_feat=shot_time_feat,
            )
            return torch.sigmoid(select_logit), torch.sigmoid(rank_logit)

        selection_scores = self.predict_shot_scores(
            frame_repr=frame_repr,
            overlaps=overlaps,
            shot_lengths=shot_lengths,
            head='selection',
        )
        if self.shot_head_mode == 'dual':
            rank_scores = self.predict_shot_scores(
                frame_repr=frame_repr,
                overlaps=overlaps,
                shot_lengths=shot_lengths,
                head='rank',
            )
        else:
            rank_scores = selection_scores
        return selection_scores, rank_scores
