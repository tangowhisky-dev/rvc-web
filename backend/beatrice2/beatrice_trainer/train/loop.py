# prepare_training() + main training loop.
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

from ..config import (AttrDict, repo_root, prepare_training_configs,
                      prepare_training_configs_for_experiment, dict_default_hparams,
                      PARAPHERNALIA_VERSION, is_notebook)
from ..io import dump_params, dump_layer
from ..layers import (CausalConv1d, WSConv1d, WSLinear, CrossAttention,
                      ConvNeXtBlock, ConvNeXtStack, VectorQuantizer)
from ..models import (
    FeatureExtractor, FeatureProjection, PhoneExtractor,
    extract_pitch_features, PitchEstimator,
    overlap_add, generate_noise, interp, linear_smoothing, dc_correction,
    nuttall, get_windowed_waveform, get_centroid, get_static_centroid,
    get_smoothed_power_spec, get_static_group_delay, get_coarse_aperiodicity,
    d4c_love_train, d4c_general_body, d4c, Vocoder, compute_loudness, slice_segments,
    ConverterNetwork,
    _normalize, SANConv2d, get_padding, DiscriminatorP, DiscriminatorR,
    MultiPeriodDiscriminator,
)
from ..train.loss import GradBalancer
from ..train.evaluation import QualityTester
from ..train.checkpoint import (get_compressed_optimizer_state_dict,
                                 get_decompressed_optimizer_state_dict)
from ..data.augment import (compute_grad_norm, compute_mean_f0, get_resampler,
                             convolve, random_formant_shift, random_filter,
                             get_noise, get_butterworth_lpf, augment_audio)
from ..data.dataset import WavDataset
from ..data.filelist import AUDIO_FILE_SUFFIXES

def prepare_training():
    # 各種準備をする
    # 副作用として、出力ディレクトリと TensorBoard のログファイルなどが生成される

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")

    # AMP: only stable on CUDA. MPS fp16 AMP has overflow issues (same as RVC
    # finding); CPU has no benefit. Disable for non-CUDA backends.
    _amp_device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    (h, in_wav_dataset_dir, out_dir, resume, skip_training, db_path, profile_id) = (
        prepare_training_configs_for_experiment
        if is_notebook()
        else prepare_training_configs
    )()

    print("config:")
    pprint(h)
    print()
    h = AttrDict(h)

    if not in_wav_dataset_dir.is_dir():
        raise ValueError(f"{in_wav_dataset_dir} is not found.")
    if resume:
        latest_checkpoint_file = out_dir / "checkpoint_latest.pt.gz"
        if not latest_checkpoint_file.is_file():
            raise ValueError(f"{latest_checkpoint_file} is not found.")
    else:
        if out_dir.is_dir():
            if (out_dir / "checkpoint_latest.pt.gz").is_file():
                raise ValueError(
                    f"{out_dir / 'checkpoint_latest.pt.gz'} already exists. "
                    "Please specify a different output directory, or use --resume option."
                )
            for file in out_dir.iterdir():
                if file.suffix == ".pt.gz":
                    raise ValueError(
                        f"{out_dir} already contains model files. "
                        "Please specify a different output directory."
                    )
        else:
            out_dir.mkdir(parents=True)

    in_ir_wav_dir = repo_root() / h.in_ir_wav_dir
    in_noise_wav_dir = repo_root() / h.in_noise_wav_dir
    in_test_wav_dir = repo_root() / h.in_test_wav_dir

    assert in_wav_dataset_dir.is_dir(), in_wav_dataset_dir
    assert out_dir.is_dir(), out_dir
    assert in_ir_wav_dir.is_dir(), in_ir_wav_dir
    assert in_noise_wav_dir.is_dir(), in_noise_wav_dir
    assert in_test_wav_dir.is_dir(), in_test_wav_dir

    # .wav または *.flac のファイルを再帰的に取得
    noise_files = sorted(
        list(in_noise_wav_dir.rglob("*.wav")) + list(in_noise_wav_dir.rglob("*.flac"))
    )
    if len(noise_files) == 0:
        raise ValueError(f"No audio data found in {in_noise_wav_dir}.")
    ir_files = sorted(
        list(in_ir_wav_dir.rglob("*.wav")) + list(in_ir_wav_dir.rglob("*.flac"))
    )
    if len(ir_files) == 0:
        raise ValueError(f"No audio data found in {in_ir_wav_dir}.")

    # TODO: 無音除去とか

    def get_training_filelist(in_wav_dataset_dir: Path):
        min_data_per_speaker = 1
        speakers: list[str] = []
        training_filelist: list[tuple[Path, int]] = []
        speaker_audio_files: list[list[Path]] = []
        for speaker_dir in sorted(in_wav_dataset_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            candidates = []
            for wav_file in sorted(speaker_dir.rglob("*")):
                if (
                    not wav_file.is_file()
                    or wav_file.suffix.lower() not in AUDIO_FILE_SUFFIXES
                    or "_originals" in wav_file.parts
                ):
                    continue
                candidates.append(wav_file)
            if len(candidates) >= min_data_per_speaker:
                speaker_id = len(speakers)
                speakers.append(speaker_dir.name)
                training_filelist.extend([(file, speaker_id) for file in candidates])
                speaker_audio_files.append(candidates)
        return speakers, training_filelist, speaker_audio_files

    speakers, training_filelist, speaker_audio_files = get_training_filelist(
        in_wav_dataset_dir
    )
    n_speakers = len(speakers)
    if n_speakers == 0:
        raise ValueError(f"No speaker data found in {in_wav_dataset_dir}.")
    print(f"{n_speakers=}")
    for i, speaker in enumerate(speakers):
        print(f"  {i:{len(str(n_speakers - 1))}d}: {speaker}")
    print()
    print(f"{len(training_filelist)=}")

    def get_test_filelist(
        in_test_wav_dir: Path, n_speakers: int
    ) -> list[tuple[Path, list[int]]]:
        max_n_test_files = 1000
        test_filelist = []
        rng = Random(42)

        def get_target_id_generator():
            if n_speakers > 8:
                while True:
                    order = list(range(n_speakers))
                    rng.shuffle(order)
                    yield from order
            else:
                while True:
                    yield from range(n_speakers)

        target_id_generator = get_target_id_generator()
        for file in sorted(in_test_wav_dir.iterdir())[:max_n_test_files]:
            if file.suffix.lower() not in AUDIO_FILE_SUFFIXES:
                continue
            target_ids = [next(target_id_generator) for _ in range(min(8, n_speakers))]
            test_filelist.append((file, target_ids))
        return test_filelist

    test_filelist = get_test_filelist(in_test_wav_dir, n_speakers)
    if len(test_filelist) == 0:
        warnings.warn(f"No audio data found in {test_filelist}.")
    print(f"{len(test_filelist)=}")
    for file, target_ids in test_filelist[:12]:
        print(f"  {file}, {target_ids}")
    if len(test_filelist) > 12:
        print("  ...")
    print()

    # データ

    training_dataset = WavDataset(
        training_filelist,
        in_sample_rate=h.in_sample_rate,
        out_sample_rate=h.out_sample_rate,
        wav_length=h.wav_length,
        segment_length=h.segment_length,
        noise_files=noise_files,
        ir_files=ir_files,
        augmentation_snr_candidates=h.augmentation_snr_candidates,
        augmentation_formant_shift_probability=h.augmentation_formant_shift_probability,
        augmentation_formant_shift_semitone_min=h.augmentation_formant_shift_semitone_min,
        augmentation_formant_shift_semitone_max=h.augmentation_formant_shift_semitone_max,
        augmentation_reverb_probability=h.augmentation_reverb_probability,
        augmentation_lpf_probability=h.augmentation_lpf_probability,
        augmentation_lpf_cutoff_freq_candidates=h.augmentation_lpf_cutoff_freq_candidates,
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        num_workers=min(h.num_workers, os.cpu_count()),
        collate_fn=training_dataset.collate,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    print("Computing mean F0s of target speakers...", end="")
    speaker_f0s = []
    for speaker, files in enumerate(speaker_audio_files):
        if len(files) > 10:
            files = Random(42).sample(files, 10)
        f0 = compute_mean_f0(files)
        speaker_f0s.append(f0)
        if speaker % 5 == 0:
            print()
        print(f"  {speaker:3d}: {f0:.1f}Hz", end=",")
    print()
    print("Done.")
    print("Computing pitch shifts for test files...")
    test_pitch_shifts = []
    source_f0s = []
    for i, (file, target_ids) in enumerate(
        tqdm(test_filelist, desc="Computing pitch shifts")
    ):
        source_f0 = compute_mean_f0([file], method="harvest")
        source_f0s.append(source_f0)
        if math.isnan(source_f0):
            test_pitch_shifts.append([0] * len(target_ids))
            continue
        pitch_shifts = []
        for target_id in target_ids:
            target_f0 = speaker_f0s[target_id]
            if target_f0 != target_f0:
                pitch_shift = 0
            else:
                pitch_shift = int(round(12.0 * math.log2(target_f0 / source_f0)))
            pitch_shifts.append(pitch_shift)
        test_pitch_shifts.append(pitch_shifts)
    print("Done.")

    # モデルと最適化

    phone_extractor = PhoneExtractor().to(device).eval().requires_grad_(False)
    phone_extractor_checkpoint = torch.load(
        repo_root() / h.phone_extractor_file, map_location="cpu", weights_only=True
    )
    print(
        phone_extractor.load_state_dict(
            phone_extractor_checkpoint["phone_extractor"], strict=False
        )
    )
    del phone_extractor_checkpoint

    pitch_estimator = PitchEstimator().to(device).eval().requires_grad_(False)
    pitch_estimator_checkpoint = torch.load(
        repo_root() / h.pitch_estimator_file, map_location="cpu", weights_only=True
    )
    print(
        pitch_estimator.load_state_dict(pitch_estimator_checkpoint["pitch_estimator"])
    )
    del pitch_estimator_checkpoint

    net_g = ConverterNetwork(
        phone_extractor,
        pitch_estimator,
        n_speakers,
        h.pitch_bins,
        h.hidden_channels,
        h.vq_topk,
        h.training_time_vq,
        h.phone_noise_ratio,
        h.floor_noise_level,
    ).to(device)
    net_d = MultiPeriodDiscriminator(san=h.san).to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        h.learning_rate_g,
        betas=h.adam_betas,
        eps=h.adam_eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        h.learning_rate_d,
        betas=h.adam_betas,
        eps=h.adam_eps,
    )

    # use_amp only when CUDA is present; MPS/CPU run fp32 (see KNOWLEDGE.md)
    _use_amp = h.use_amp and torch.cuda.is_available()
    grad_scaler = torch.amp.GradScaler(_amp_device, enabled=_use_amp)
    grad_balancer = GradBalancer(
        weights={
            "loss_loudness": h.grad_weight_loudness,
            "loss_mel": h.grad_weight_mel,
            "loss_adv": h.grad_weight_adv,
            "loss_fm": h.grad_weight_fm,
        }
        | ({"loss_ap": h.grad_weight_ap} if h.grad_weight_ap else {}),
        ema_decay=h.grad_balancer_ema_decay,
    )
    resample_to_in_sample_rate = torchaudio.transforms.Resample(
        h.out_sample_rate, h.in_sample_rate
    ).to(device)

    # チェックポイント読み出し

    initial_iteration = 0
    if resume:  # 学習再開
        checkpoint_file = latest_checkpoint_file
    elif h.pretrained_file is not None:  # ファインチューニング
        checkpoint_file = repo_root() / h.pretrained_file
    else:  # 事前学習
        checkpoint_file = None

    if checkpoint_file is not None:
        with gzip.open(checkpoint_file, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)
        if not resume and not skip_training:  # ファインチューニング
            initial_speaker_embedding = checkpoint["net_g"]["embed_speaker.weight"][:1]
            initial_speaker_embedding_for_cross_attention = checkpoint["net_g"][
                "key_value_speaker_embedding.weight"
            ][:1]
            checkpoint["net_g"]["embed_speaker.weight"] = initial_speaker_embedding[
                [0] * n_speakers
            ]
            checkpoint["net_g"]["key_value_speaker_embedding.weight"] = (
                initial_speaker_embedding_for_cross_attention[[0] * n_speakers]
            )
            checkpoint["net_g"]["vq.codebooks"] = checkpoint["net_g"]["vq.codebooks"][
                [0] * n_speakers
            ]
        print(net_g.load_state_dict(checkpoint["net_g"], strict=False))
        print(net_d.load_state_dict(checkpoint["net_d"], strict=False))
        if resume or skip_training:
            optim_g.load_state_dict(
                get_decompressed_optimizer_state_dict(checkpoint["optim_g"])
            )
            optim_d.load_state_dict(
                get_decompressed_optimizer_state_dict(checkpoint["optim_d"])
            )
            initial_iteration = checkpoint["iteration"]
        grad_balancer.load_state_dict(checkpoint["grad_balancer"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    def wav_iterator(files):
        for file in files:
            wav, sr = torchaudio.load(file)
            wav = wav.to(device)
            if sr != h.in_sample_rate:
                wav = get_resampler(sr, h.in_sample_rate, device)(wav)
            yield wav[:, None, :]

    if resume:
        net_g.enable_hook()
    else:
        net_g.initialize_vq([wav_iterator(files) for files in speaker_audio_files])

    # スケジューラ

    def get_exponential_warmup_scheduler(
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        decay: float,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return current_epoch / warmup_epochs
            else:
                return decay ** (current_epoch - warmup_epochs)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler_g = get_exponential_warmup_scheduler(
        optim_g, h.warmup_steps, h.learning_rate_decay
    )
    scheduler_d = get_exponential_warmup_scheduler(
        optim_d, h.warmup_steps, h.learning_rate_decay
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`\.",
        )
        for _ in range(initial_iteration + 1):
            scheduler_g.step()
            scheduler_d.step()

    net_g.train()
    net_d.train()

    # ログとか

    dict_scalars = defaultdict(list)
    quality_tester = QualityTester().eval().to(device)
    if skip_training:
        writer = None
    else:
        writer = SummaryWriter(out_dir, flush_secs=10)
        if not h.record_metrics:
            writer.add_scalar = lambda *args, **kwargs: None
            writer.add_histogram = lambda *args, **kwargs: None
        writer.add_text(
            "log",
            f"start training w/ {torch.cuda.get_device_name(device) if torch.cuda.is_available() else str(device)}.",
            initial_iteration,
        )
    if not resume:
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(dict(h), f, indent=4)
        if not is_notebook():
            shutil.copy(__file__, out_dir)

    return (
        device,
        in_wav_dataset_dir,
        h,
        out_dir,
        speakers,
        test_filelist,
        training_loader,
        speaker_f0s,
        test_pitch_shifts,
        phone_extractor,
        pitch_estimator,
        net_g,
        net_d,
        optim_g,
        optim_d,
        grad_scaler,
        grad_balancer,
        resample_to_in_sample_rate,
        initial_iteration,
        scheduler_g,
        scheduler_d,
        dict_scalars,
        quality_tester,
        writer,
        _amp_device,
        _use_amp,
        db_path,
        profile_id,
    )



def _run_main():
    """Entry point called by __main__.py."""
    (
        device,
        in_wav_dataset_dir,
        h,
        out_dir,
        speakers,
        test_filelist,
        training_loader,
        speaker_f0s,
        test_pitch_shifts,
        phone_extractor,
        pitch_estimator,
        net_g,
        net_d,
        optim_g,
        optim_d,
        grad_scaler,
        grad_balancer,
        resample_to_in_sample_rate,
        initial_iteration,
        scheduler_g,
        scheduler_d,
        dict_scalars,
        quality_tester,
        writer,
        _amp_device,
        _use_amp,
        db_path,
        profile_id,
    ) = prepare_training()

    # Direct DB writer — same pattern as RVC's TrainingDBWriter
    from backend.beatrice2.db_writer import BeatriceDBWriter
    _db_writer = BeatriceDBWriter(db_path, profile_id)

    # Best-UTMOS tracking — checkpoint_best.pt.gz copied in save block
    _best_utmos: float = -float("inf")
    _best_utmos_step: int = -1
    _needs_best_copy: bool = False  # set when UTMOS improves; cleared after save block copies

    if h.compile_convnext:
        raw_convnextstack_forward = ConvNeXtStack.forward
        compiled_convnextstack_forward = torch.compile(
            ConvNeXtStack.forward, mode="reduce-overhead"
        )
    if h.compile_d4c:
        d4c = torch.compile(d4c, mode="reduce-overhead")
    if h.compile_discriminator:
        MultiPeriodDiscriminator.forward_and_compute_loss = torch.compile(
            MultiPeriodDiscriminator.forward_and_compute_loss, mode="reduce-overhead"
        )

    # 学習
    with (
        torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1500, warmup=10, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
        )
        if h.profile
        else nullcontext()
    ) as profiler:
        data_iter = iter(training_loader)
        for iteration in tqdm(range(initial_iteration, h.n_steps), desc="Training"):
            # === 1. データ前処理 ===
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(training_loader)
                batch = next(data_iter)
            (
                clean_wavs,
                noisy_wavs_16k,
                slice_starts,
                speaker_ids,
                formant_shift_semitone,
            ) = map(lambda x: x.to(device, non_blocking=True), batch)

            # === 2. 学習 ===
            with torch.amp.autocast(_amp_device, enabled=_use_amp):
                # === 2.1 Generator の順伝播 ===
                if h.compile_convnext:
                    ConvNeXtStack.forward = compiled_convnextstack_forward
                (
                    y,
                    y_hat,
                    y_hat_for_backward,
                    loss_loudness,
                    loss_mel,
                    loss_ap,
                    generator_stats,
                ) = net_g.forward_and_compute_loss(
                    noisy_wavs_16k[:, None, :],
                    speaker_ids,
                    formant_shift_semitone,
                    slice_start_indices=slice_starts,
                    slice_segment_length=h.segment_length,
                    y_all=clean_wavs[:, None, :],
                    enable_loss_ap=h.grad_weight_ap != 0.0,
                )
                if h.compile_convnext:
                    ConvNeXtStack.forward = raw_convnextstack_forward
                assert y_hat.isfinite().all()
                assert loss_loudness.isfinite().all()
                assert loss_mel.isfinite().all()
                assert loss_ap.isfinite().all()

                # === 2.2 Discriminator の順伝播 ===
                loss_discriminator, loss_adv, loss_fm, discriminator_stats = (
                    net_d.forward_and_compute_loss(y, y_hat)
                )
                assert loss_discriminator.isfinite().all()
                assert loss_adv.isfinite().all()
                assert loss_fm.isfinite().all()

            # === 2.3 Discriminator の逆伝播 ===
            for param in net_d.parameters():
                assert param.grad is None
            grad_scaler.scale(loss_discriminator).backward(
                retain_graph=True, inputs=list(net_d.parameters())
            )
            loss_discriminator = loss_discriminator.item()
            grad_scaler.unscale_(optim_d)
            if iteration % 5 == 0:
                grad_norm_d, d_grad_norm_stats = compute_grad_norm(net_d, True)
            else:
                grad_norm_d = math.nan
                d_grad_norm_stats = {}

            # === 2.4 Generator の逆伝播 ===
            for param in net_g.parameters():
                assert param.grad is None
            gradient_balancer_stats = grad_balancer.backward(
                {
                    "loss_loudness": loss_loudness,
                    "loss_mel": loss_mel,
                    "loss_adv": loss_adv,
                    "loss_fm": loss_fm,
                }
                | ({"loss_ap": loss_ap} if h.grad_weight_ap else {}),
                y_hat_for_backward,
                grad_scaler,
                skip_update_ema=iteration > 10 and iteration % 5 != 0,
            )
            loss_loudness = loss_loudness.item()
            loss_mel = loss_mel.item()
            loss_adv = loss_adv.item()
            loss_fm = loss_fm.item()
            if h.grad_weight_ap:
                loss_ap = loss_ap.item()
            grad_scaler.unscale_(optim_g)
            if iteration % 5 == 0:
                grad_norm_g, g_grad_norm_stats = compute_grad_norm(net_g, True)
            else:
                grad_norm_g = math.nan
                g_grad_norm_stats = {}

            # === 2.5 パラメータの更新 ===
            grad_scaler.step(optim_g)
            optim_g.zero_grad(set_to_none=True)
            grad_scaler.step(optim_d)
            optim_d.zero_grad(set_to_none=True)
            grad_scaler.update()

            # === 3. ログ ===
            dict_scalars["loss_g/loss_loudness"].append(loss_loudness)
            dict_scalars["loss_g/loss_mel"].append(loss_mel)
            if h.grad_weight_ap:
                dict_scalars["loss_g/loss_ap"].append(loss_ap)
            dict_scalars["loss_g/loss_fm"].append(loss_fm)
            dict_scalars["loss_g/loss_adv"].append(loss_adv)
            dict_scalars["other/grad_scale"].append(grad_scaler.get_scale())
            dict_scalars["loss_d/loss_discriminator"].append(loss_discriminator)
            if math.isfinite(grad_norm_d):
                dict_scalars["other/gradient_norm_d"].append(grad_norm_d)
                for name, value in d_grad_norm_stats.items():
                    dict_scalars[f"~gradient_norm_d/{name}"].append(value)
            if math.isfinite(grad_norm_g):
                dict_scalars["other/gradient_norm_g"].append(grad_norm_g)
                for name, value in g_grad_norm_stats.items():
                    dict_scalars[f"~gradient_norm_g/{name}"].append(value)
            dict_scalars["other/lr_g"].append(scheduler_g.get_last_lr()[0])
            dict_scalars["other/lr_d"].append(scheduler_d.get_last_lr()[0])
            for k, v in generator_stats.items():
                dict_scalars[f"~loss_generator/{k}"].append(v)
            for k, v in discriminator_stats.items():
                dict_scalars[f"~loss_discriminator/{k}"].append(v)
            for k, v in gradient_balancer_stats.items():
                dict_scalars[f"~gradient_balancer/{k}"].append(v)
            _tb_interval = 10
            if (iteration + 1) % _tb_interval == 0 or iteration == 0:
                # Compute averages before clearing
                scalar_avgs = {
                    name: sum(vals) / len(vals)
                    for name, vals in dict_scalars.items()
                    if vals
                }
                for name, avg in scalar_avgs.items():
                    writer.add_scalar(name, avg, iteration + 1)
                    dict_scalars[name].clear()
                writer.flush()

                # Write step losses directly to DB — same pattern as RVC's TrainingDBWriter
                _db_writer.insert_step_loss(iteration + 1, {
                    "loss_mel":  scalar_avgs.get("loss_g/loss_mel"),
                    "loss_loud": scalar_avgs.get("loss_g/loss_loudness"),
                    "loss_adv":  scalar_avgs.get("loss_g/loss_adv"),
                    "loss_fm":   scalar_avgs.get("loss_g/loss_fm"),
                    "loss_ap":   scalar_avgs.get("loss_g/loss_ap"),
                    "loss_d":    scalar_avgs.get("loss_d/loss_discriminator"),
                })

                # Histograms are expensive — only every 1000 iters
                if (iteration + 1) % 1000 == 0 or iteration == 0:
                    for name, param in net_g.named_parameters():
                        writer.add_histogram(f"weight/{name}", param, iteration + 1)

                intermediate_feature_stats = {}
                hook_handles = []

                def get_layer_hook(name):
                    def compute_stats(module, x, suffix):
                        if not isinstance(x, torch.Tensor):
                            return
                        if x.dtype not in [torch.float32, torch.float16]:
                            return
                        if isinstance(module, nn.Identity):
                            return
                        x = x.detach().float()
                        var = x.var().item()
                        if isinstance(module, (nn.Linear, nn.LayerNorm)):
                            channel_var, channel_mean = torch.var_mean(
                                x.reshape(-1, x.size(-1)), 0
                            )
                        elif isinstance(module, nn.Conv1d):
                            channel_var, channel_mean = torch.var_mean(x, [0, 2])
                        else:
                            return
                        average_squared_channel_mean = (
                            channel_mean.square().mean().item()
                        )
                        average_channel_var = channel_var.mean().item()

                        tensor_idx = len(intermediate_feature_stats) // 3
                        intermediate_feature_stats[
                            f"var/{tensor_idx:02d}_{name}/{suffix}"
                        ] = var
                        intermediate_feature_stats[
                            f"avg_sq_ch_mean/{tensor_idx:02d}_{name}/{suffix}"
                        ] = average_squared_channel_mean
                        intermediate_feature_stats[
                            f"avg_ch_var/{tensor_idx:02d}_{name}/{suffix}"
                        ] = average_channel_var

                    def forward_pre_hook(module, input):
                        for i, input_i in enumerate(input):
                            compute_stats(module, input_i, f"input_{i}")

                    def forward_hook(module, input, output):
                        if isinstance(output, tuple):
                            for i, output_i in enumerate(output):
                                compute_stats(module, output_i, f"output_{i}")
                        else:
                            compute_stats(module, output, "output")

                    return forward_pre_hook, forward_hook

                for name, layer in net_g.named_modules():
                    forward_pre_hook, forward_hook = get_layer_hook(name)
                    hook_handles.append(
                        layer.register_forward_pre_hook(forward_pre_hook)
                    )
                    hook_handles.append(layer.register_forward_hook(forward_hook))
                with torch.no_grad(), torch.amp.autocast(_amp_device, enabled=_use_amp):
                    net_g.forward_and_compute_loss(
                        noisy_wavs_16k[:, None, :],
                        speaker_ids,
                        formant_shift_semitone,
                        slice_start_indices=slice_starts,
                        slice_segment_length=h.segment_length,
                        y_all=clean_wavs[:, None, :],
                        enable_loss_ap=h.grad_weight_ap != 0.0,
                    )
                for handle in hook_handles:
                    handle.remove()
                for name, value in intermediate_feature_stats.items():
                    writer.add_scalar(
                        f"~intermediate_feature_{name}", value, iteration + 1
                    )

            # === 4. 検証 ===
            if (iteration + 1) % h.evaluation_interval == 0 or iteration + 1 in {
                1,
                h.n_steps,
            }:
                torch.backends.cudnn.benchmark = False
                net_g.eval()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                dict_qualities_all = defaultdict(list)
                n_added_wavs = 0
                with torch.inference_mode():
                    for i, ((file, target_ids), pitch_shift_semitones) in enumerate(
                        zip(test_filelist, test_pitch_shifts)
                    ):
                        source_wav, sr = torchaudio.load(file)
                        source_wav = source_wav.to(device)
                        if sr != h.in_sample_rate:
                            source_wav = get_resampler(sr, h.in_sample_rate, device)(
                                source_wav
                            )
                        source_wav = source_wav.to(device)
                        original_source_wav_length = source_wav.size(1)
                        # 長さのパターンを減らしてキャッシュを効かせる
                        if source_wav.size(1) % h.in_sample_rate == 0:
                            padded_source_wav = source_wav
                        else:
                            padded_source_wav = F.pad(
                                source_wav,
                                (
                                    0,
                                    h.in_sample_rate
                                    - source_wav.size(1) % h.in_sample_rate,
                                ),
                            )
                        converted = net_g(
                            padded_source_wav[[0] * len(target_ids), None],
                            torch.tensor(target_ids, device=device),
                            torch.tensor(
                                [0.0] * len(target_ids), device=device
                            ),  # フォルマントシフト
                            torch.tensor(
                                [float(p) for p in pitch_shift_semitones], device=device
                            ),
                        ).squeeze_(1)[:, : original_source_wav_length // 160 * 240]
                        if i < 12:
                            if iteration == 0:
                                writer.add_audio(
                                    f"source/y_{i:02d}",
                                    source_wav,
                                    iteration + 1,
                                    h.in_sample_rate,
                                )
                            for d in range(
                                min(
                                    len(target_ids),
                                    1 + (12 - i - 1) // len(test_filelist),
                                )
                            ):
                                idx_in_batch = n_added_wavs % len(target_ids)
                                writer.add_audio(
                                    f"converted/y_hat_{i:02d}_{target_ids[idx_in_batch]:03d}_{pitch_shift_semitones[idx_in_batch]:+02d}",
                                    converted[idx_in_batch],
                                    iteration + 1,
                                    h.out_sample_rate,
                                )
                                n_added_wavs += 1
                        converted = resample_to_in_sample_rate(converted)
                        quality = quality_tester.test(converted, source_wav)
                        for metric_name, values in quality.items():
                            dict_qualities_all[metric_name].extend(values)
                assert n_added_wavs == min(
                    12, len(test_filelist) * len(test_filelist[0][1])
                ), (
                    n_added_wavs,
                    len(test_filelist),
                    len(speakers),
                    len(test_filelist[0][1]),
                )
                dict_qualities = {
                    metric_name: sum(values) / len(values)
                    for metric_name, values in dict_qualities_all.items()
                    if len(values)
                }
                for metric_name, value in dict_qualities.items():
                    writer.add_scalar(f"validation/{metric_name}", value, iteration + 1)
                for metric_name, values in dict_qualities_all.items():
                    for i, value in enumerate(values):
                        writer.add_scalar(
                            f"~validation_{metric_name}/{i:03d}", value, iteration + 1
                        )

                # Write UTMOS to DB and track best step — checkpoint copied in save block below
                _utmos_now = dict_qualities.get("utmos")
                if _utmos_now is not None:
                    _warmup = (iteration + 1) <= 1000
                    _is_new_best = (not _warmup) and (_utmos_now > _best_utmos)
                    _db_writer.insert_step_loss(iteration + 1, {"utmos": _utmos_now, "is_best": 1 if _is_new_best else 0})
                    if _is_new_best:
                        _best_utmos = _utmos_now
                        _best_utmos_step = iteration + 1
                        _needs_best_copy = True
                        _db_writer.update_best_utmos(iteration + 1, _utmos_now)
                        print(f"[best] new best UTMOS {_utmos_now:.4f} at step {iteration + 1}", flush=True)

                del dict_qualities, dict_qualities_all

                net_g.train()
                torch.backends.cudnn.benchmark = True
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # === 5. 保存 ===
            if (iteration + 1) % h.save_interval == 0 or iteration + 1 in {
                1,
                h.n_steps,
            }:
                # チェックポイント
                name = f"{in_wav_dataset_dir.name}_{iteration + 1:08d}"
                checkpoint_file_save = out_dir / f"checkpoint_{name}.pt.gz"
                if checkpoint_file_save.exists():
                    checkpoint_file_save = checkpoint_file_save.with_name(
                        f"{checkpoint_file_save.name}_{hash(None):x}"
                    )
                with gzip.open(checkpoint_file_save, "wb") as f:
                    torch.save(
                        {
                            "iteration": iteration + 1,
                            "net_g": net_g.state_dict(),
                            "phone_extractor": phone_extractor.state_dict(),
                            "pitch_estimator": pitch_estimator.state_dict(),
                            "net_d": {
                                k: v.half() for k, v in net_d.state_dict().items()
                            },
                            "optim_g": get_compressed_optimizer_state_dict(optim_g),
                            "optim_d": get_compressed_optimizer_state_dict(optim_d),
                            "grad_balancer": grad_balancer.state_dict(),
                            "grad_scaler": grad_scaler.state_dict(),
                            "h": dict(h),
                        },
                        f,
                    )
                shutil.copy(checkpoint_file_save, out_dir / "checkpoint_latest.pt.gz")
                # During warmup (≤1000 steps), latest IS best — keep checkpoint_best in sync.
                # After warmup, only copy when UTMOS genuinely improves.
                best_path = out_dir / "checkpoint_best.pt.gz"
                if _needs_best_copy:
                    shutil.copy(checkpoint_file_save, best_path)
                    _needs_best_copy = False
                    print(f"[best] checkpoint_best.pt.gz updated (step {_best_utmos_step}, UTMOS {_best_utmos:.4f})", flush=True)
                elif (iteration + 1) <= 1000:
                    shutil.copy(checkpoint_file_save, best_path)

                # 推論用
                paraphernalia_dir = out_dir / f"paraphernalia_{name}"
                if paraphernalia_dir.exists():
                    paraphernalia_dir = paraphernalia_dir.with_name(
                        f"{paraphernalia_dir.name}_{hash(None):x}"
                    )
                paraphernalia_dir.mkdir()
                phone_extractor_fp16 = PhoneExtractor()
                phone_extractor_fp16.load_state_dict(phone_extractor.state_dict())
                phone_extractor_fp16.remove_weight_norm()
                phone_extractor_fp16.merge_weights()
                phone_extractor_fp16.half()
                phone_extractor_fp16.dump(paraphernalia_dir / "phone_extractor.bin")
                del phone_extractor_fp16
                pitch_estimator_fp16 = PitchEstimator()
                pitch_estimator_fp16.load_state_dict(pitch_estimator.state_dict())
                pitch_estimator_fp16.merge_weights()
                pitch_estimator_fp16.half()
                pitch_estimator_fp16.dump(paraphernalia_dir / "pitch_estimator.bin")
                del pitch_estimator_fp16
                net_g_fp16 = ConverterNetwork(
                    nn.Module(),
                    nn.Module(),
                    len(speakers),
                    h.pitch_bins,
                    h.hidden_channels,
                    h.vq_topk,
                    h.training_time_vq,
                    h.phone_noise_ratio,
                    h.floor_noise_level,
                )
                net_g_fp16.load_state_dict(net_g.state_dict())
                net_g_fp16.merge_weights()
                net_g_fp16.half()
                net_g_fp16.dump(paraphernalia_dir / "waveform_generator.bin")
                net_g_fp16.dump_speaker_embeddings(
                    paraphernalia_dir / "speaker_embeddings.bin"
                )
                net_g_fp16.dump_embedding_setter(
                    paraphernalia_dir / "embedding_setter.bin"
                )
                del net_g_fp16
                _noimage_src = repo_root() / "assets/images/noimage.png"
                if not _noimage_src.exists():
                    _noimage_src.parent.mkdir(parents=True, exist_ok=True)
                    # Create a minimal 1x1 white PNG (no external deps needed)
                    import struct, zlib
                    def _make_1x1_png():
                        def chunk(name, data):
                            c = struct.pack(">I", len(data)) + name + data
                            return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
                        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
                        idat = zlib.compress(b"\x00\xFF\xFF\xFF")
                        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
                    _noimage_src.write_bytes(_make_1x1_png())
                shutil.copy(
                    _noimage_src, paraphernalia_dir
                )
                with open(
                    paraphernalia_dir / f"beatrice_paraphernalia_{name}.toml",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        f'''[model]
    version = "{PARAPHERNALIA_VERSION}"
    name = "{name}"
    description = """
    No description for this model.
    このモデルの説明はありません。
    """
    '''
                    )
                    for speaker_id, (speaker, speaker_f0) in enumerate(
                        zip(speakers, speaker_f0s)
                    ):
                        average_pitch = 69.0 + 12.0 * math.log2(speaker_f0 / 440.0)
                        average_pitch = round(average_pitch * 8.0) / 8.0
                        f.write(
                            f'''
    [voice.{speaker_id}]
    name = "{speaker}"
    description = """
    No description for this voice.
    この声の説明はありません。
    """
    average_pitch = {average_pitch}

    [voice.{speaker_id}.portrait]
    path = "noimage.png"
    description = """
    """
    '''
                        )
                del paraphernalia_dir

            # TODO: phone_extractor, pitch_estimator が既知のモデルであれば dump を省略

            # === 6. スケジューラ更新 ===
            scheduler_g.step()
            scheduler_d.step()
            if h.profile:
                profiler.step()

    print("Training finished.")
