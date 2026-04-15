# Hyperparameter defaults, repo-root resolution, arg parsing.
import argparse
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

def is_notebook() -> bool:
    return "get_ipython" in globals()


def repo_root() -> Path:
    if "BEATRICE_TRAINER_ROOT" in os.environ:
        return Path(os.environ["BEATRICE_TRAINER_ROOT"])
    d = Path.cwd() / "dummy" if is_notebook() else Path(__file__)
    assert d.is_absolute(), d
    for d in d.parents:
        if (d / ".git").is_dir():
            return d
    raise RuntimeError("Repository root is not found.")


# ハイパーパラメータ
# 学習データや出力ディレクトリなど、学習ごとに変わるようなものはここに含めない
dict_default_hparams = {
    # training
    "learning_rate_g": 5e-5,
    "learning_rate_d": 5e-5,
    "learning_rate_decay": 0.999999,
    "adam_betas": [0.8, 0.99],
    "adam_eps": 1e-6,
    "batch_size": 8,
    "grad_weight_loudness": 1.0,  # grad_weight は比が同じなら同じ意味になるはず
    "grad_weight_mel": 50.0,
    "grad_weight_ap": 100.0,
    "grad_weight_adv": 150.0,
    "grad_weight_fm": 150.0,
    "grad_balancer_ema_decay": 0.995,
    "use_amp": True,
    "num_workers": 32,
    "n_steps": 10000,
    "warmup_steps": 5000,
    "evaluation_interval": 2000,
    "save_interval": 2000,
    "in_sample_rate": 16000,  # 変更不可
    "out_sample_rate": 24000,  # 変更不可
    "wav_length": 4 * 24000,  # 4s
    "segment_length": 100,  # 1s
    "phone_noise_ratio": 0.5,
    "vq_topk": 4,
    "training_time_vq": "none",  # "none", "self" or "random"
    "floor_noise_level": 1e-3,
    "record_metrics": True,
    # augmentation
    "augmentation_snr_candidates": [20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
    "augmentation_formant_shift_probability": 0.5,
    "augmentation_formant_shift_semitone_min": -3.0,
    "augmentation_formant_shift_semitone_max": 3.0,
    "augmentation_reverb_probability": 0.5,
    "augmentation_lpf_probability": 0.2,
    "augmentation_lpf_cutoff_freq_candidates": [2000.0, 3000.0, 4000.0, 6000.0],
    # data
    "phone_extractor_file": "assets/beatrice2/pretrained/122_checkpoint_03000000.pt",
    "pitch_estimator_file": "assets/beatrice2/pretrained/104_3_checkpoint_00300000.pt",
    "in_ir_wav_dir": "assets/beatrice2/ir",
    "in_noise_wav_dir": "assets/beatrice2/noise",
    "in_test_wav_dir": "assets/beatrice2/test",
    "pretrained_file": "assets/beatrice2/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz",  # None も可
    # model
    "pitch_bins": 448,  # 変更不可
    "hidden_channels": 256,  # ファインチューン時変更不可、変更した場合は推論側の対応必要
    "san": False,  # ファインチューン時変更不可
    "compile_convnext": False,
    "compile_d4c": False,
    "compile_discriminator": False,
    "profile": False,
}

if __name__ == "__main__":
    # スクリプト内部のデフォルト設定と assets/default_config.json が同期されているか確認
    default_config_file = repo_root() / "assets/default_config.json"
    if default_config_file.is_file():
        with open(default_config_file, encoding="utf-8") as f:
            default_config: dict = json.load(f)
        for key, value in dict_default_hparams.items():
            if key not in default_config:
                warnings.warn(f"{key} not found in default_config.json.")
            else:
                if value != default_config[key]:
                    warnings.warn(
                        f"{key} differs between default_config.json ({default_config[key]}) and internal default hparams ({value})."
                    )
                del default_config[key]
        for key in default_config:
            warnings.warn(f"{key} found in default_config.json is unknown.")
    else:
        warnings.warn("dafualt_config.json not found.")


def prepare_training_configs_for_experiment() -> tuple[dict, Path, Path, bool, bool]:
    import ipynbname  # type: ignore[import]
    from IPython import get_ipython  # type: ignore[import]

    h = deepcopy(dict_default_hparams)
    in_wav_dataset_dir = repo_root() / "../../data/processed/libritts_r_200"
    try:
        notebook_name = ipynbname.name()
    except FileNotFoundError:
        notebook_name = Path(get_ipython().user_ns["__vsc_ipynb_file__"]).name
    out_dir = repo_root() / "notebooks" / notebook_name.split(".")[0].split("_")[0]
    resume = False
    skip_training = False
    return h, in_wav_dataset_dir, out_dir, resume, skip_training


def prepare_training_configs() -> tuple[dict, Path, Path, bool, bool]:
    # data_dir, out_dir は config ファイルでもコマンドライン引数でも指定でき、
    # コマンドライン引数が優先される。
    # 各種ファイルパスを相対パスで指定した場合、config ファイルでは
    # リポジトリルートからの相対パスとなるが、コマンドライン引数では
    # カレントディレクトリからの相対パスとなる。

    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-d", "--data_dir", type=Path, help="directory containing the training data")
    parser.add_argument("-o", "--out_dir", type=Path, help="output directory")
    parser.add_argument("-r", "--resume", action="store_true", help="resume training")
    parser.add_argument("-c", "--config", type=Path, help="path to the config file")
    # fmt: on
    args = parser.parse_args()

    # config
    if args.config is None:
        h = deepcopy(dict_default_hparams)
    else:
        with open(args.config, encoding="utf-8") as f:
            h = json.load(f)
    for key in dict_default_hparams.keys():
        if key not in h:
            h[key] = dict_default_hparams[key]
            warnings.warn(
                f"{key} is not specified in the config file. Using the default value."
            )
    # data_dir
    if args.data_dir is not None:
        in_wav_dataset_dir = args.data_dir
    elif "data_dir" in h:
        in_wav_dataset_dir = repo_root() / Path(h["data_dir"])
        del h["data_dir"]
    else:
        raise ValueError(
            "data_dir must be specified. "
            "For example `python3 beatrice_trainer -d my_training_data_dir -o my_output_dir`."
        )
    # out_dir
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif "out_dir" in h:
        out_dir = repo_root() / Path(h["out_dir"])
        del h["out_dir"]
    else:
        raise ValueError(
            "out_dir must be specified. "
            "For example `python3 beatrice_trainer -d my_training_data_dir -o my_output_dir`."
        )
    for key in list(h.keys()):
        if key not in dict_default_hparams:
            warnings.warn(f"`{key}` specified in the config file will be ignored.")
            del h[key]
    # resume
    resume = args.resume
    return h, in_wav_dataset_dir, out_dir, resume, False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# %% [markdown]
# ## Phone Extractor


