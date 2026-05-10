import torch
from torch import nn


class BudgetedSigmoid(nn.Module):
    """Length-aware continuous budget selector.

    p_i = sigmoid((select_logit_i - lambda * length_i) / tau)

    lambda is found by bisection on detached logits so the returned mask remains
    differentiable with respect to select_logits while satisfying the target
    budget as closely as possible.

    Shape contract:
        select_logits: [S]
        lengths: [S], frame counts such as nfps
        budget_frames: scalar frame budget
        return: soft_mask [S], lambda scalar
    """

    def __init__(self, tau: float = 0.2, max_iter: int = 50, eps: float = 1e-6):
        super().__init__()
        if tau <= 0:
            raise ValueError(f'Invalid tau={tau}; expected > 0.')
        if max_iter <= 0:
            raise ValueError(f'Invalid max_iter={max_iter}; expected > 0.')
        self.tau = float(tau)
        self.max_iter = int(max_iter)
        self.eps = float(eps)

    def forward(self,
                select_logits: torch.Tensor,
                lengths: torch.Tensor,
                budget_frames) -> tuple:
        if select_logits.ndim != 1:
            raise ValueError(f'Expected select_logits shape [S], got {tuple(select_logits.shape)}')
        if lengths.ndim != 1:
            raise ValueError(f'Expected lengths shape [S], got {tuple(lengths.shape)}')
        if select_logits.shape[0] != lengths.shape[0]:
            raise ValueError(
                f'select_logits/lengths mismatch: {select_logits.shape[0]} vs {lengths.shape[0]}'
            )
        if not torch.isfinite(select_logits).all():
            raise ValueError('Non-finite select_logits in BudgetedSigmoid.')
        if not torch.isfinite(lengths).all():
            raise ValueError('Non-finite lengths in BudgetedSigmoid.')
        lengths = lengths.to(device=select_logits.device, dtype=select_logits.dtype).clamp_min(self.eps)
        total_length = lengths.sum()
        target = torch.as_tensor(
            budget_frames,
            device=select_logits.device,
            dtype=select_logits.dtype,
        ).clamp(min=0.0, max=float(total_length.detach().item()))

        if float(target.detach().item()) <= self.eps:
            return torch.zeros_like(select_logits), select_logits.new_tensor(float('inf'))
        if float((total_length - target).detach().item()) <= self.eps:
            return torch.ones_like(select_logits), select_logits.new_tensor(float('-inf'))

        detached_logits = select_logits.detach()
        detached_lengths = lengths.detach()
        detached_target = target.detach()

        def selected_length(lam: torch.Tensor) -> torch.Tensor:
            probs = torch.sigmoid((detached_logits - lam * detached_lengths) / self.tau)
            return torch.sum(probs * detached_lengths)

        low = select_logits.new_tensor(-1.0)
        high = select_logits.new_tensor(1.0)
        while float(selected_length(low).item()) < float(detached_target.item()):
            low = low * 2.0
        while float(selected_length(high).item()) > float(detached_target.item()):
            high = high * 2.0

        for _ in range(self.max_iter):
            mid = (low + high) * 0.5
            if float(selected_length(mid).item()) > float(detached_target.item()):
                low = mid
            else:
                high = mid
        lam = ((low + high) * 0.5).detach()
        soft_mask = torch.sigmoid((select_logits - lam * lengths) / self.tau)
        return soft_mask, lam
