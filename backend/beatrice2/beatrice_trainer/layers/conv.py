# Causal and weight-standardised 1-D conv primitives.
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

class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        delay: int = 0,
    ):
        padding = (kernel_size - 1) * dilation - delay
        self.trim = (kernel_size - 1) * dilation - 2 * delay
        if self.trim < 0:
            raise ValueError
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super().forward(input)
        if self.trim == 0:
            return result
        else:
            return result[:, :, : -self.trim]


class WSConv1d(CausalConv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        delay: int = 0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            delay=delay,
        )
        self.weight.data.normal_(
            0.0, math.sqrt(1.0 / (in_channels * kernel_size // groups))
        )
        if bias:
            self.bias.data.zero_()
        self.gain = nn.Parameter(torch.ones((out_channels, 1, 1)))

    def standardized_weight(self) -> torch.Tensor:
        var, mean = torch.var_mean(self.weight, [1, 2], keepdim=True)
        scale = (
            self.gain
            * (
                self.in_channels * self.kernel_size[0] // self.groups * var + 1e-8
            ).rsqrt()
        )
        return scale * (self.weight - mean)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.conv1d(
            input,
            self.standardized_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.trim == 0:
            return result
        else:
            return result[:, :, : -self.trim]

    def merge_weights(self):
        self.weight.data[:] = self.standardized_weight().detach()
        self.gain.data.fill_(1.0)


class WSLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.weight.data.normal_(0.0, math.sqrt(1.0 / in_features))
        self.bias.data.zero_()
        self.gain = nn.Parameter(torch.ones((out_features, 1)))

    def standardized_weight(self) -> torch.Tensor:
        var, mean = torch.var_mean(self.weight, 1, keepdim=True)
        scale = self.gain * (self.in_features * var + 1e-8).rsqrt()
        return scale * (self.weight - mean)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.standardized_weight(), self.bias)

    def merge_weights(self):
        self.weight.data[:] = self.standardized_weight().detach()
        self.gain.data.fill_(1.0)


