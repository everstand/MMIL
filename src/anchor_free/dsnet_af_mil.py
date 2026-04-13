from torch import nn
import torch

from modules.models import build_base_model


class DSNetAFMIL(nn.Module):
    def __init__(self,
                 base_model: str,
                 num_feature: int,
                 num_hidden: int,
                 num_head: int,
                 num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.base_model = build_base_model(base_model, num_feature, num_head)
        self.layer_norm = nn.LayerNorm(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(num_hidden),
        )

        self.fc_cls = nn.Linear(num_hidden, num_classes)
        self.fc_attn = nn.Linear(num_hidden, 1)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise ValueError(f'Expected input shape [B, T, D], got {tuple(x.shape)}')
        if x.shape[0] != 1:
            raise ValueError(
                f'DSNetAFMIL expects batch size 1 in current training pipeline, got {x.shape[0]}'
            )

        _, seq_len, _ = x.shape

        out = self.base_model(x)
        out = out + x
        out = self.layer_norm(out)
        out = self.fc1(out)

        instance_logits = self.fc_cls(out).view(seq_len, self.num_classes)
        attn_logits = self.fc_attn(out).view(seq_len)
        attn_weights = torch.softmax(attn_logits, dim=0)

        bag_logits = torch.sum(
            attn_weights.unsqueeze(-1) * instance_logits,
            dim=0,
        )

        return instance_logits, attn_logits, attn_weights, bag_logits

    @torch.no_grad()
    def predict_summary_scores(self, seq: torch.Tensor) -> torch.Tensor:
        _, _, attn_weights, _ = self(seq)
        return attn_weights