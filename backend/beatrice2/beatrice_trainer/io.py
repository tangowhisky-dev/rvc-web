# Shared binary serialisation helpers used across all model modules.
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

def dump_params(params: torch.Tensor, f: BinaryIO):
    if params is None:
        return
    if params.dtype == torch.bfloat16:
        f.write(
            params.detach()
            .clone()
            .float()
            .view(torch.short)
            .numpy()
            .ravel()[1::2]
            .tobytes()
        )
    else:
        f.write(params.detach().numpy().ravel().tobytes())
    f.flush()


def dump_layer(layer: nn.Module, f: BinaryIO):
    dump = partial(dump_params, f=f)
    if hasattr(layer, "dump"):
        layer.dump(f)
    elif isinstance(layer, (nn.Linear, nn.Conv1d, nn.LayerNorm)):
        dump(layer.weight)
        dump(layer.bias)
    elif isinstance(layer, nn.MultiheadAttention):
        embed_dim = layer.embed_dim
        num_heads = layer.num_heads
        # [3 * embed_dim, embed_dim]
        in_proj_weight = layer.in_proj_weight.data.clone()
        in_proj_weight[: 2 * embed_dim] *= 1.0 / math.sqrt(
            math.sqrt(embed_dim // num_heads)
        )
        in_proj_weight = in_proj_weight.view(
            3, num_heads, embed_dim // num_heads, embed_dim
        )
        # [num_heads, 3, embed_dim / num_heads, embed_dim]
        in_proj_weight = in_proj_weight.transpose(0, 1)
        # [3 * embed_dim]
        in_proj_bias = layer.in_proj_bias.data.clone()
        in_proj_bias[: 2 * embed_dim] *= 1.0 / math.sqrt(
            math.sqrt(embed_dim // num_heads)
        )
        in_proj_bias = in_proj_bias.view(3, num_heads, embed_dim // num_heads)
        # [num_heads, 3, embed_dim / num_heads]
        in_proj_bias = in_proj_bias.transpose(0, 1)
        dump(in_proj_weight)
        dump(in_proj_bias)
        dump(layer.out_proj.weight)
        dump(layer.out_proj.bias)
    elif isinstance(layer, nn.Embedding):
        dump(layer.weight)
    elif isinstance(layer, nn.Parameter):
        dump(layer)
    elif isinstance(layer, nn.ModuleList):
        for layer_i in layer:
            dump_layer(layer_i, f)
    else:
        assert False, layer


