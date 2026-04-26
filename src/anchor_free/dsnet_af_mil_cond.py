from torch import nn
import torch

from modules.models import build_base_model


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
                 score_head: str = 'single'):
        super().__init__()

        if score_head not in ('single', 'dual', 'residual_dual'):
            raise ValueError(
                f'Invalid score_head={score_head}; expected single, dual, or residual_dual.'
            )

        self.num_classes = num_classes
        self.num_feature = num_feature
        self.score_head = score_head

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

    def forward(self, x: torch.Tensor, text_cond: torch.Tensor):
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

        raw_x = x

        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)

        pre_cross_frame_repr = out.squeeze(0)

        cond_out, _ = self.cross_attn(
            query=out,
            key=text_cond,
            value=text_cond,
            need_weights=False,
        )
        cond_out = self.cross_attn_layer_norm(cond_out + out)

        hidden = self.fc1(cond_out).squeeze(0)
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
        )

    @torch.no_grad()
    def predict_summary_scores(self, seq: torch.Tensor, text_cond: torch.Tensor) -> torch.Tensor:
        _, _, summary_scores, _, _, _ = self(seq, text_cond)
        return summary_scores
