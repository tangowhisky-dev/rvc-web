# QualityTester (UTMOS MOS scoring).
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

class QualityTester(nn.Module):
    def __init__(self):
        super().__init__()
        self.utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.0.0", "utmos22_strong", trust_repo=True
        ).eval()

    @torch.inference_mode()
    def compute_mos(self, wav: torch.Tensor) -> dict[str, list[float]]:
        res = {"utmos": self.utmos(wav, sr=16000).tolist()}
        return res

    def test(
        self, converted_wav: torch.Tensor, source_wav: torch.Tensor
    ) -> dict[str, list[float]]:
        # [batch_size, wav_length]
        res = {}
        res.update(self.compute_mos(converted_wav))
        return res

    def test_many(
        self, converted_wavs: list[torch.Tensor], source_wavs: list[torch.Tensor]
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        # list[batch_size, wav_length]
        results = defaultdict(list)
        assert len(converted_wavs) == len(source_wavs)
        for converted_wav, source_wav in zip(converted_wavs, source_wavs):
            res = self.test(converted_wav, source_wav)
            for metric_name, value in res.items():
                results[metric_name].extend(value)
        return {
            metric_name: sum(values) / len(values)
            for metric_name, values in results.items()
        }, results


