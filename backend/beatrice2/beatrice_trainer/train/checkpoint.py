# Checkpoint compression helpers.
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

def get_compressed_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
) -> dict:
    state_dict = {}
    for k0, v0 in optimizer.state_dict().items():
        if k0 != "state":
            state_dict[k0] = v0
            continue
        state_dict[k0] = {}
        for k1, v1 in v0.items():
            state_dict[k0][k1] = {}
            for k2, v2 in v1.items():
                if isinstance(v2, torch.Tensor):
                    state_dict[k0][k1][k2] = v2.bfloat16()
                    assert state_dict[k0][k1][k2].isfinite().all()
                else:
                    state_dict[k0][k1][k2] = v2
    return state_dict


def get_decompressed_optimizer_state_dict(compressed_state_dict: dict) -> dict:
    state_dict = {}
    for k0, v0 in compressed_state_dict.items():
        if k0 != "state":
            state_dict[k0] = v0
            continue
        state_dict[k0] = {}
        for k1, v1 in v0.items():
            state_dict[k0][k1] = {}
            for k2, v2 in v1.items():
                if isinstance(v2, torch.Tensor):
                    state_dict[k0][k1][k2] = v2.float()
                    assert state_dict[k0][k1][k2].isfinite().all()
                else:
                    state_dict[k0][k1][k2] = v2
    return state_dict


