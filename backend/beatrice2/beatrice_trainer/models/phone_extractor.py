# FeatureExtractor, FeatureProjection, PhoneExtractor.
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
from ..layers.conv import WSConv1d
from ..layers.attention import ConvNeXtStack
from ..io import dump_params, dump_layer

class FeatureExtractor(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        # fmt: off
        self.conv0 = weight_norm(nn.Conv1d(1, hidden_channels // 8, 10, 5, bias=False))
        self.conv1 = weight_norm(nn.Conv1d(hidden_channels // 8, hidden_channels // 4, 3, 2, bias=False))
        self.conv2 = weight_norm(nn.Conv1d(hidden_channels // 4, hidden_channels // 2, 3, 2, bias=False))
        self.conv3 = weight_norm(nn.Conv1d(hidden_channels // 2, hidden_channels, 3, 2, bias=False))
        self.conv4 = weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, 2, bias=False))
        self.conv5 = weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 2, 2, bias=False))
        # fmt: on

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, wav_length]
        wav_length = x.size(2)
        if wav_length % 160 != 0:
            warnings.warn("wav_length % 160 != 0")
        x = F.pad(x, (40, 40))
        x = F.gelu(self.conv0(x), approximate="tanh")
        x = F.gelu(self.conv1(x), approximate="tanh")
        x = F.gelu(self.conv2(x), approximate="tanh")
        x = F.gelu(self.conv3(x), approximate="tanh")
        x = F.gelu(self.conv4(x), approximate="tanh")
        x = F.gelu(self.conv5(x), approximate="tanh")
        # [batch_size, hidden_channels, wav_length / 160]
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv0)
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)
        remove_weight_norm(self.conv3)
        remove_weight_norm(self.conv4)
        remove_weight_norm(self.conv5)

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.conv0, f)
        dump_layer(self.conv1, f)
        dump_layer(self.conv2, f)
        dump_layer(self.conv3, f)
        dump_layer(self.conv4, f)
        dump_layer(self.conv5, f)


class FeatureProjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch_size, channels, length]
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        return x


class PhoneExtractor(nn.Module):
    def __init__(
        self,
        phone_channels: int = 128,
        hidden_channels: int = 128,
        backbone_embed_kernel_size: int = 9,
        kernel_size: int = 17,
        n_blocks: int = 20,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(hidden_channels)
        self.feature_projection = FeatureProjection(hidden_channels)
        self.backbone = ConvNeXtStack(
            in_channels=hidden_channels,
            channels=hidden_channels,
            intermediate_channels=hidden_channels * 3,
            n_blocks=n_blocks,
            delay=0,
            embed_kernel_size=backbone_embed_kernel_size,
            kernel_size=kernel_size,
            use_mha=True,
        )
        self.head = weight_norm(nn.Conv1d(hidden_channels, phone_channels, 1))

    def forward(
        self, x: torch.Tensor, return_stats: bool = True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, float]]]:
        # x: [batch_size, 1, wav_length]

        stats = {}

        # [batch_size, 1, wav_length] -> [batch_size, feature_extractor_hidden_channels, length]
        x = self.feature_extractor(x)
        if return_stats:
            stats["feature_norm"] = x.detach().norm(dim=1).mean()
        # [batch_size, feature_extractor_hidden_channels, length] -> [batch_size, hidden_channels, length]
        x = self.feature_projection(x)
        # [batch_size, hidden_channels, length]
        x = self.backbone(x)
        # [batch_size, hidden_channels, length] -> [batch_size, phone_channels, length]
        phone = self.head(F.gelu(x, approximate="tanh"))

        results = [phone]
        if return_stats:
            stats["code_norm"] = phone.detach().norm(dim=1).mean()
            results.append(stats)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    @torch.inference_mode()
    def units(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 1, wav_length]

        # [batch_size, 1, wav_length] -> [batch_size, phone_channels, length]
        phone = self.forward(x, return_stats=False)
        # [batch_size, phone_channels, length] -> [batch_size, length, phone_channels]
        phone = phone.transpose(1, 2)
        # [batch_size, length, phone_channels]
        return phone

    def remove_weight_norm(self):
        self.feature_extractor.remove_weight_norm()
        remove_weight_norm(self.head)

    def merge_weights(self):
        self.backbone.merge_weights()

        self.backbone.embed.bias.data += (
            (
                self.feature_projection.norm.bias.data[None, :, None]
                * self.backbone.embed.weight.data  # [o, i, k]
            )
            .sum(1)
            .sum(1)
        )
        self.backbone.embed.weight.data *= self.feature_projection.norm.weight.data[
            None, :, None
        ]
        self.feature_projection.norm.bias.data[:] = 0.0
        self.feature_projection.norm.weight.data[:] = 1.0

    def dump(self, f: Union[BinaryIO, str, bytes, os.PathLike]):
        if isinstance(f, (str, bytes, os.PathLike)):
            with open(f, "wb") as f:
                self.dump(f)
            return
        if not hasattr(f, "write"):
            raise TypeError

        dump_layer(self.feature_extractor, f)
        dump_layer(self.backbone, f)
        dump_layer(self.head, f)


