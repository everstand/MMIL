from torch import nn
import torch

from modules.models import build_base_model


class DSNetAFMILCond(nn.Module):
    def __init__(self,
                 base_model: str,
                 num_feature: int,
                 num_hidden: int,
                 num_head: int,
                 num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_feature = num_feature

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

    def forward(self,
                x: torch.Tensor,
                text_cond: torch.Tensor):
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

        base_frame_repr = out.squeeze(0)

        cond_out, _ = self.cross_attn(
            query=out,
            key=text_cond,
            value=text_cond,
            need_weights=False,
        )
        cond_out = self.cross_attn_layer_norm(cond_out + out)

        hidden = self.fc1(cond_out).squeeze(0)
        base_frame_repr = cond_out.squeeze(0)
        raw_frame_features = raw_x.squeeze(0)

        instance_logits = self.fc_cls(hidden)
        attn_logits = self.fc_attn(hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=0)

        bag_logits = torch.sum(
            attn_weights.unsqueeze(-1) * instance_logits,
            dim=0,
        )

        summary_feat = torch.sum(
            attn_weights.unsqueeze(-1) * raw_frame_features,
            dim=0,
        )

        return (
            instance_logits,
            attn_logits,
            attn_weights,
            bag_logits,
            summary_feat,
            base_frame_repr,
        )

    @torch.no_grad()
    def predict_summary_scores(self,
                               seq: torch.Tensor,
                               text_cond: torch.Tensor) -> torch.Tensor:
        _, _, attn_weights, _, _, _ = self(seq, text_cond)
        return attn_weights