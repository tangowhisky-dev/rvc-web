# GradBalancer.
import gc
import gzip
import json
import math
import os
import shutil
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from fractions import Fraction
from functools import partial
from pathlib import Path
from pprint import pprint
from random import Random
from typing import BinaryIO, Literal, Optional, Union, Sequence, Iterable, Callable

import numpy as np
import pyworld
import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    assert "soundfile" in torchaudio.list_audio_backends()
except AttributeError:
    import soundfile as _sf  # noqa: F401

if not hasattr(torch.amp, "GradScaler"):
    class GradScaler(torch.cuda.amp.GradScaler):
        def __init__(self, _, *args, **kwargs):
            super().__init__(*args, **kwargs)
    torch.amp.GradScaler = GradScaler

PARAPHERNALIA_VERSION = "2.0.0-rc.0"

class GradBalancer:
    """Adapted from https://github.com/facebookresearch/encodec/blob/main/encodec/balancer.py"""

    def __init__(
        self,
        weights: dict[str, float],
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
    ):
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.ema_decay = ema_decay
        self.rescale_grads = rescale_grads

        self.ema_total: dict[str, float] = defaultdict(float)
        self.ema_fix: dict[str, float] = defaultdict(float)

    def backward(
        self,
        losses: dict[str, torch.Tensor],
        input: torch.Tensor,
        scaler: Optional[torch.amp.GradScaler] = None,
        skip_update_ema: bool = False,
    ) -> dict[str, float]:
        stats = {}
        if skip_update_ema:
            assert len(losses) == len(self.ema_total)
            ema_norms = {k: tot / self.ema_fix[k] for k, tot in self.ema_total.items()}
        else:
            # 各 loss に対して d loss / d input とそのノルムを計算する
            norms = {}
            grads = {}
            for name, loss in losses.items():
                if scaler is not None:
                    loss = scaler.scale(loss)
                (grad,) = torch.autograd.grad(loss, [input], retain_graph=True)
                if not grad.isfinite().all():
                    input.backward(grad)
                    return {}
                grad = grad.detach() / (1.0 if scaler is None else scaler.get_scale())
                if self.per_batch_item:
                    dims = tuple(range(1, grad.dim()))
                    ema_norm = grad.norm(dim=dims).mean()
                else:
                    ema_norm = grad.norm()
                norms[name] = float(ema_norm)
                grads[name] = grad

            # ノルムの移動平均を計算する
            for key, value in norms.items():
                self.ema_total[key] = self.ema_total[key] * self.ema_decay + value
                self.ema_fix[key] = self.ema_fix[key] * self.ema_decay + 1.0
            ema_norms = {k: tot / self.ema_fix[k] for k, tot in self.ema_total.items()}

            # ログを取る
            total_ema_norm = sum(ema_norms.values())
            for k, ema_norm in ema_norms.items():
                stats[f"grad_norm_value_{k}"] = ema_norm
                stats[f"grad_norm_ratio_{k}"] = ema_norm / (total_ema_norm + 1e-12)

        # loss の係数の比率を計算する
        if self.rescale_grads:
            total_weights = sum([self.weights[k] for k in ema_norms])
            ratios = {k: w / total_weights for k, w in self.weights.items()}

        # 勾配を修正する
        loss = 0.0
        for name, ema_norm in ema_norms.items():
            if self.rescale_grads:
                scale = ratios[name] * self.total_norm / (ema_norm + 1e-12)
            else:
                scale = self.weights[name]
            loss += (losses if skip_update_ema else grads)[name] * scale
        if scaler is not None:
            loss = scaler.scale(loss)
        if skip_update_ema:
            (loss,) = torch.autograd.grad(loss, [input])
        input.backward(loss)
        return stats

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {
            "ema_total": dict(self.ema_total),
            "ema_fix": dict(self.ema_fix),
        }

    def load_state_dict(self, state_dict):
        self.ema_total = defaultdict(float, state_dict["ema_total"])
        self.ema_fix = defaultdict(float, state_dict["ema_fix"])


