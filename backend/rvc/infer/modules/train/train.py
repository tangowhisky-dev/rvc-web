import os
import sys
import logging
from typing import Tuple

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle
import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from rvc.ipex import ipex_init, gradscaler_init
        from torch.xpu.amp import autocast

        GradScaler = gradscaler_init()
        ipex_init()
    else:
        # Use the device-agnostic torch.amp API (torch.cuda.amp is deprecated in PyTorch 2.x)
        from torch.amp import GradScaler, autocast
except Exception:
    from torch.amp import GradScaler, autocast

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

from rvc.layers.discriminators import MultiPeriodDiscriminator
from infer.lib.train.mel_processing import MultiScaleMelSpectrogramLoss

if hps.version == "v1":
    from rvc.layers.synthesizers import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from rvc.layers.synthesizers import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from rvc.layers.synthesizers import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
    )
from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    speaker_loss as _spk_loss_fn,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import save_small_model

from rvc.layers.utils import (
    slice_on_last_dim,
    total_grad_norm,
)

# ---------------------------------------------------------------------------
# Speaker encoder (ECAPA-TDNN) — loaded once at module level if c_spk > 0
# ---------------------------------------------------------------------------
_speaker_encoder = None


def _get_speaker_encoder(device: torch.device):
    """Lazy-load the ECAPA-TDNN speaker encoder."""
    global _speaker_encoder
    if _speaker_encoder is None:
        from infer.lib.train.speaker_encoder import SpeakerEncoder

        project_root = os.environ.get(
            "PROJECT_ROOT",
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
            ),
        )
        pt_path = os.path.join(project_root, "assets", "ecapa", "ecapa_tdnn.pt")
        _speaker_encoder = SpeakerEncoder(pt_path=pt_path, device=device)
    return _speaker_encoder


global_step = 0
lowest_loss_g = float("inf")  # best epoch-average generator loss seen so far


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()


def run(rank, n_gpus, hps: utils.HParams, logger: logging.Logger):
    global global_step
    if rank == 0:
        # logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # Determine compute device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        _device = torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    _use_mps = _device.type == "mps"

    # Init process group for distributed training (CUDA multi-GPU only).
    # Skip for MPS: DDP collective ops not implemented for MPS, and single-device
    # MPS doesn't need distributed gradient sync.
    if not _use_mps:
        try:
            dist.init_process_group(
                backend=(
                    "gloo"
                    if os.name == "nt" or not torch.cuda.is_available()
                    else "nccl"
                ),
                init_method="env://",
                world_size=n_gpus,
                rank=rank,
            )
        except:
            dist.init_process_group(
                backend=(
                    "gloo"
                    if os.name == "nt" or not torch.cuda.is_available()
                    else "nccl"
                ),
                init_method="env://?use_libuv=False",
                world_size=n_gpus,
                rank=rank,
            )

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    # DataLoader workers re-import train.py (argparse at module-level makes num_workers>0
    # crash under mp.spawn on macOS). Use num_workers=0 for MPS/CPU.
    # MPS device placement (below) is the real performance win — not worker parallelism.
    _use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4 if _use_cuda else 0,
        shuffle=False,
        pin_memory=_use_cuda,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=_use_cuda,
        prefetch_factor=8 if _use_cuda else None,
    )
    mdl = hps.copy().model
    del mdl.use_spectral_norm
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **mdl,
            sr=hps.sample_rate,
            vocoder=getattr(hps, "vocoder", "HiFi-GAN"),
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **mdl,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    elif _use_mps:
        net_g = net_g.to(_device)
    has_xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    _vocoder = getattr(hps, "vocoder", "HiFi-GAN")
    _disc_version = "v3" if _vocoder == "RefineGAN" else hps.version
    net_d = MultiPeriodDiscriminator(
        _disc_version,
        use_spectral_norm=hps.model.use_spectral_norm,
        has_xpu=has_xpu,
    )
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    elif _use_mps:
        net_d = net_d.to(_device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        pass
    elif torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    elif _use_mps:
        # Skip DDP on MPS — allgather/allreduce collectives aren't implemented for MPS.
        # Single-device MPS training doesn't need distributed gradient sync at all.
        pass  # net_g and net_d stay as plain modules on MPS device
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        # epoch_str is the epoch that was just *saved* (completed). Advance by
        # one so the loop starts at the next unfinished epoch.
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(
                            hps.pretrainG, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(
                            hps.pretrainG, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(
                            hps.pretrainD, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(
                            hps.pretrainD, map_location="cpu", weights_only=True
                        )["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(_device.type, enabled=hps.train.fp16_run)

    # Mel loss function — multiscale for RefineGAN, L1 for HiFi-GAN
    _use_multiscale_mel = getattr(hps, "vocoder", "HiFi-GAN") == "RefineGAN"
    if _use_multiscale_mel:
        _fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=hps.data.sampling_rate)
    else:
        _fn_mel_loss = None  # use F.l1_loss directly

    # Speaker encoder + profile embedding — loaded once if c_spk > 0
    _spk_encoder = None
    _profile_emb = None
    _c_spk = getattr(hps.train, "c_spk", 0.0)
    if _c_spk > 0 and rank == 0:
        try:
            _spk_encoder = _get_speaker_encoder(_device)
            logger.info(f"Speaker encoder loaded (c_spk={_c_spk})")

            # Load profile-level speaker embedding
            profile_emb_path = os.path.join(hps.model_dir, "profile_embedding.pt")
            if os.path.exists(profile_emb_path):
                prof_data = torch.load(
                    profile_emb_path, map_location=_device, weights_only=False
                )
                _profile_emb = prof_data["profile_embedding"]  # [192]
                logger.info(
                    f"Profile embedding loaded: norm={_profile_emb.norm().item():.4f}"
                )
            else:
                logger.warning(
                    f"Profile embedding not found at {profile_emb_path} — speaker loss disabled"
                )
                _c_spk = 0.0
        except Exception as e:
            logger.warning(
                f"Failed to load speaker encoder/profile embedding: {e} — speaker loss disabled"
            )
            _c_spk = 0.0

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
                _device,
                _use_mps,
                _fn_mel_loss,
                _spk_encoder,
                _c_spk,
                _profile_emb,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
                _device,
                _use_mps,
                _fn_mel_loss,
                None,
                0.0,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets: Tuple[RVC_Model_f0, MultiPeriodDiscriminator],
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
    cache,
    _device=None,
    _use_mps=False,
    fn_mel_loss=None,
    spk_encoder=None,
    c_spk=0.0,
    profile_emb=None,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    global lowest_loss_g

    # Per-epoch loss accumulation for best-epoch detection
    _epoch_gen_sum: float = 0.0
    _epoch_gen_count: int = 0

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on device (CUDA or MPS)
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                elif _use_mps:
                    phone = phone.to(_device)
                    phone_lengths = phone_lengths.to(_device)
                    if hps.if_f0 == 1:
                        pitch = pitch.to(_device)
                        pitchf = pitchf.to(_device)
                    sid = sid.to(_device)
                    spec = spec.to(_device)
                    spec_lengths = spec_lengths.to(_device)
                    wave = wave.to(_device)
                    wave_lengths = wave_lengths.to(_device)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        pitch = pitchf = None
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on device (CUDA or MPS)
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
        elif (hps.if_cache_data_in_gpu == False) and _use_mps:
            phone = phone.to(_device)
            phone_lengths = phone_lengths.to(_device)
            if hps.if_f0 == 1:
                pitch = pitch.to(_device)
                pitchf = pitchf.to(_device)
            sid = sid.to(_device)
            spec = spec.to(_device)
            spec_lengths = spec_lengths.to(_device)
            wave = wave.to(_device)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(_device.type, enabled=hps.train.fp16_run):
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(phone, phone_lengths, spec, spec_lengths, sid, pitch, pitchf)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = slice_on_last_dim(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(_device.type, enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = slice_on_last_dim(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(_device.type, enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = total_grad_norm(net_d.parameters())
        scaler.step(optim_d)

        with autocast(_device.type, enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(_device.type, enabled=False):
                if fn_mel_loss is not None:
                    # RefineGAN: multi-scale mel loss on raw waveform slices
                    loss_mel = fn_mel_loss(wave, y_hat) * hps.train.c_mel / 3.0
                else:
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # Speaker loss — profile-level embedding vs generated segment
                loss_spk = torch.tensor(0.0, device=_device)
                if spk_encoder is not None and c_spk > 0 and profile_emb is not None:
                    # Resample to 16kHz for ECAPA-TDNN
                    _sr = hps.data.sampling_rate
                    # wave and y_hat are [B, 1, T] — squeeze to [B, T]
                    _yhat = y_hat.squeeze(1)
                    if _sr != 16000:
                        _ratio = 16000 / _sr
                        _new_len = int(_yhat.shape[-1] * _ratio)
                        _yhat_16k = F.interpolate(
                            _yhat.unsqueeze(1),
                            size=_new_len,
                            mode="linear",
                            align_corners=False,
                        ).squeeze(1)
                    else:
                        _yhat_16k = _yhat

                    gen_emb = spk_encoder.get_embedding(_yhat_16k)  # [B, 192]
                    # Normalize embeddings for cosine similarity
                    gen_emb = F.normalize(gen_emb, p=2, dim=1)
                    ref_emb = profile_emb.unsqueeze(
                        0
                    )  # [1, 192] → [B, 192] via broadcast
                    loss_spk = _spk_loss_fn(gen_emb, ref_emb) * c_spk

                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_spk
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = total_grad_norm(net_g.parameters())
        scaler.step(optim_g)
        scaler.update()

        # Accumulate generator loss for epoch-average best-epoch tracking
        if rank == 0:
            _epoch_gen_sum += loss_gen.item()
            _epoch_gen_count += 1

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"losses :_disc={loss_disc:.3f}, _gen={loss_gen:.3f}, _fm={loss_fm:.3f},_mel={loss_mel:.3f}, _kl={loss_kl:.3f}, _spk={loss_spk:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                        "loss/g/spk": loss_spk,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_latest.pth"),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_latest.pth"),
            )
        # Release fragmented MPS allocator blocks back to the system after
        # each checkpoint save. On macOS unified memory, the MPS allocator
        # holds onto free blocks by default; flushing here keeps the memory
        # footprint stable across long runs without impacting step latency.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    save_small_model(
                        ckpt,
                        hps.sample_rate,
                        hps.if_f0,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        hps.version,
                        hps,
                    ),
                )
            )

    if rank == 0:
        # Best-epoch checkpoint — save G_best.pth whenever epoch-average
        # generator loss hits a new minimum (mirrors ultimate-rvc strategy).
        # This runs every epoch regardless of save_every_epoch so the best
        # checkpoint is always captured at the exact right epoch, not the
        # nearest save_every_epoch boundary.
        if _epoch_gen_count > 0:
            avg_loss_g = _epoch_gen_sum / _epoch_gen_count
            _min_delta = 0.004  # must improve by at least this to count
            if avg_loss_g < lowest_loss_g - _min_delta:
                lowest_loss_g = avg_loss_g
                best_path = os.path.join(hps.model_dir, "G_best.pth")
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    best_path,
                )
                logger.info(
                    f"best model updated → G_best.pth  (epoch={epoch}, avg_loss_g={avg_loss_g:.4f})"
                )

        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                save_small_model(
                    ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps
                )
            )
        )
        sleep(1)
        # Flush all logging handlers before os._exit() bypasses normal cleanup
        logging.shutdown()
        os._exit(0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
