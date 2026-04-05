"""AdamSPD — Adam with Selective Projection Decay.

Applies a stiffness penalty that projects weights back toward their
pretrained anchor values only when the gradient direction would move
parameters further from the anchor.  This limits catastrophic forgetting
during fine-tuning without a hard constraint on weight magnitude.

Based on the implementation from codename-rvc-fork-4:
  rvc/train/custom_optimizers/adamspd/adamSPD.py

Usage in train.py:
    import copy
    anchors_g = [p.data.clone() for p in net_g.parameters() if p.requires_grad]
    param_groups_g = [{'params': [p for p in net_g.parameters() if p.requires_grad],
                       'pre': anchors_g}]
    optim_g = AdamSPD(param_groups_g, lr=lr, betas=(0.8,0.99), eps=1e-9, weight_decay=0.5)

weight_decay acts as the stiffness multiplier (projection strength).
Typical value: 0.01–0.5.  Higher = stronger pull toward pretrain.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class AdamSPD(Optimizer):
    """Adam with Selective Projection Decay.

    Parameters
    ----------
    params : iterable
        Parameter groups.  Each group must contain a ``'pre'`` key holding
        a list of anchor tensors (same order as ``'params'``).  Obtain
        anchors by ``[p.data.clone() for p in model.parameters() if p.requires_grad]``
        before training starts.
    lr : float
    betas : (float, float)
    eps : float
    weight_decay : float
        Projection strength — how hard the optimizer pulls parameters back
        toward their pretrained values.  0 = pure Adam (no penalty).
    amsgrad : bool
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.8, 0.99),
        eps: float = 1e-9,
        weight_decay: float = 0.1,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, pre=None)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @staticmethod
    def _spd_ratio(new_p: torch.Tensor, param: torch.Tensor,
                   pre: torch.Tensor) -> torch.Tensor:
        """Projection ratio clamped to [0, 1]."""
        curr_norm = torch.norm(new_p - pre)
        prev_norm = torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-12)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            anchors = group.get("pre")  # list of anchor tensors, same order as params

            anchor_idx = 0
            for param in group["params"]:
                if param.grad is None:
                    if anchors is not None:
                        anchor_idx += 1
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamSPD does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom = (max_sq.sqrt() / math.sqrt(bias2)).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)

                step_size = lr / bias1
                d_p = step_size * exp_avg / denom
                new_p = param - d_p

                # Selective Projection Decay — only pull toward anchor when
                # the gradient is pushing further away from it.
                if anchors is not None and wd > 0.0:
                    pre = anchors[anchor_idx].to(param.device, non_blocking=True)
                    condition = -torch.sum(grad * (param - pre))
                    if condition < 0.0:
                        ratio = self._spd_ratio(new_p, param, pre)
                        new_p = new_p - wd * ratio * (new_p - pre)

                param.copy_(new_p)

                if anchors is not None:
                    anchor_idx += 1

        return loss
