# VectorQuantizer.
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

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_speakers: int,
        codebook_size: int,
        channels: int,
        topk: int = 4,
        training_time_vq: Literal["none", "self", "random"] = "none",
    ):
        super().__init__()
        assert 1 <= topk <= codebook_size
        self.n_speakers = n_speakers
        self.codebook_size = codebook_size
        self.channels = channels
        self.topk = topk
        self.training_time_vq = training_time_vq

        self.register_buffer(
            "codebooks",
            torch.empty(n_speakers, codebook_size, channels, dtype=torch.half),
        )
        self.codebooks: torch.Tensor

        # VQ の適用箇所を変更しやすいように hook にしている
        self._hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.target_speaker_ids: Optional[torch.Tensor] = None

        def _hook(_, __, output):
            return self(output, self.target_speaker_ids)

        self._hook_fn = _hook

    @torch.no_grad()
    def build_codebooks(
        self,
        collector_func: Callable,
        target_layer: nn.Module,
        inputs: Sequence[Iterable[torch.Tensor]],
        kmeans_n_iters: int = 50,
    ):
        assert len(inputs) == self.n_speakers
        assert self._hook_handle is None, "hook already installed"
        device = next(self.buffers()).device

        for spk_id, inps in enumerate(tqdm(inputs, desc="Building codebooks")):
            activations: list[torch.Tensor] = []

            # TODO: データ多すぎる場合に間引く処理をする

            def _collect(_, __, output):
                # output: [batch_size, channels, length]
                activations.append(output.detach())

            handle = target_layer.register_forward_hook(_collect)
            for x in inps:
                collector_func(x.to(device))
            handle.remove()

            if not activations:
                raise RuntimeError(f"No activation collected for speaker {spk_id}")

            # [n_data, channels]
            activations: torch.Tensor = torch.cat(
                [
                    a.transpose(1, 2).reshape(a.size(0) * a.size(2), self.channels)
                    for a in activations
                ]
            )
            activations = activations.float()
            activations = F.normalize(activations, dim=1, eps=1e-6)
            # [codebook_size, channels]
            centers = (
                self._kmeans_plus_plus(activations, self.codebook_size, kmeans_n_iters)
                if activations.size(0) >= self.codebook_size
                else self._pad_replicate(activations, self.codebook_size)
            )
            self.codebooks[spk_id] = centers.to(self.codebooks.dtype)

    def forward(
        self, x: torch.Tensor, speaker_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, channels, length = x.size()
        assert channels == self.channels
        device = x.device
        dtype = x.dtype

        if self.training:
            if self.training_time_vq == "none":
                return x
            elif self.training_time_vq == "self":
                if self.target_speaker_ids is None:
                    raise ValueError("target_speaker_ids is not set")
            elif self.training_time_vq == "random":
                speaker_ids = torch.randint(
                    0, self.n_speakers, (batch_size,), device=device
                )
            else:
                raise ValueError(f"Unknown training_time_vq: {self.training_time_vq}")
        else:
            if speaker_ids is None:
                return x
            speaker_ids = speaker_ids.to(device)

        # [batch_size, channels, length] → [batch_size, length, channels]
        q = F.normalize(x, dim=1, eps=1e-6)
        codes = self.codebooks[speaker_ids].to(q.dtype)
        # [batch_size, length, codebook_size]
        sim = torch.einsum("bcl,bkc->blk", q, codes)

        # [batch_size, length, topk]
        _, topk_idx = sim.topk(self.topk, dim=-1)
        # [batch_size, length, codebook_size, channels]
        expanded_codes = codes[:, None, :, :].expand(-1, length, -1, -1)
        # [batch_size, length, topk, channels]
        expanded_topk_idx = topk_idx[:, :, :, None].expand(-1, -1, -1, channels)
        # [batch_size, length, topk, channels]
        gathered = expanded_codes.gather(2, expanded_topk_idx)
        # [batch_size, length, channels]
        gathered = gathered.mean(2)
        # [batch_size, channels, length]
        return gathered.transpose(1, 2).to(dtype)

    def enable_hook(self, target_layer: nn.Module):
        if self._hook_handle is not None:
            raise RuntimeError("hook already installed")
        self._hook_handle = target_layer.register_forward_hook(self._hook_fn)

    def disable_hook(self):
        if self._hook_handle is None:
            raise RuntimeError("hook not installed")
        self._hook_handle.remove()
        self._hook_handle = None

    def set_target_speaker_ids(self, speaker_ids: Optional[torch.Tensor]):
        # この話者が使われる条件は forward() を参照
        self.target_speaker_ids = speaker_ids

    @staticmethod
    def _pad_replicate(x: torch.Tensor, n: int) -> torch.Tensor:
        # データ数が n に満たないとき適当に複製して埋める
        idx = torch.arange(n, device=x.device) % x.size(0)
        return x[idx]

    @staticmethod
    def _kmeans_plus_plus(
        x: torch.Tensor, n_clusters: int, n_iters: int = 50
    ) -> torch.Tensor:
        n_data, _ = x.size()
        center_indices = [torch.randint(0, n_data, ()).item()]
        min_distances = torch.full((n_data,), math.inf, device=x.device)
        for _ in range(1, n_clusters):
            last_center_index = center_indices[-1]
            min_distances = min_distances.minimum(
                torch.cdist(x, x[last_center_index : last_center_index + 1])
                .float()
                .square_()
                .squeeze_(1)
            )
            probs = min_distances / (min_distances.sum() + 1e-12)
            center_indices.append(torch.multinomial(probs, 1).item())
        centers = x[center_indices]
        del min_distances, probs
        for _ in range(n_iters):
            distances = torch.cdist(x, centers)  # [n_data, n_clusters]
            labels = distances.argmin(1)  # [n_data]
            # [n_clusters, dim]
            new_centers = torch.zeros_like(centers).index_add_(0, labels, x)
            # [n_clusters]
            counts = labels.bincount(minlength=n_clusters)
            if (counts == 0).sum().item() != 0:
                # TODO: 割り当てがないクラスタの処理
                warnings.warn("Some clusters have no assigned data points.")
            new_centers /= counts[:, None].clamp_(min=1).float()
            centers = new_centers
        return centers


# %% [markdown]
# ## Pitch Estimator


