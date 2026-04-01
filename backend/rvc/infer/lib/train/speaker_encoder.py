"""ECAPA-TDNN speaker encoder — pure PyTorch, zero NeMo dependency.

Exact replica of NeMo's ECAPAEncoder + SpeakerDecoder architecture.
Uses librosa for mel filterbank and torch.view_as_real for STFT magnitude
to produce bit-identical results with NeMo.

Usage:
    from infer.lib.train.speaker_encoder import SpeakerEncoder
    encoder = SpeakerEncoder(pt_path="assets/ecapa/ecapa_tdnn.pt", device="cpu")
    emb = encoder.get_embedding(audio)  # audio: [B, T] at 16kHz
"""

import os
import math
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper functions (exact NeMo replicas)
# ---------------------------------------------------------------------------


def _get_same_padding(kernel_size, stride=1, dilation=1):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def _lens_to_mask(lens, max_len, device=None):
    lens_mat = torch.arange(max_len, device=device)
    mask = lens_mat[:max_len].unsqueeze(0) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1)
    num_values = torch.sum(mask, dim=2, keepdim=True)
    return mask, num_values


def _get_statistics_with_mask(x, m, dim=2, eps=1e-10):
    mean = torch.sum((m * x), dim=dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std


def _normalize_batch(x, seq_len, normalize_type="per_feature"):
    if normalize_type == "per_feature":
        if seq_len is not None:
            # Fully vectorised — no Python loop, no .item() GPU→CPU sync stalls.
            # Works correctly on CUDA, MPS, and CPU.
            max_t  = x.shape[-1]
            # mask: [B, T]  —  True for valid time steps
            mask   = torch.arange(max_t, device=x.device).unsqueeze(0) < seq_len.unsqueeze(1)
            mask_f = mask.unsqueeze(1).float()              # [B, 1, T]
            counts = mask_f.sum(dim=-1).clamp(min=1)        # [B, 1]
            x_mean = (x * mask_f).sum(dim=-1) / counts      # [B, C]
            diff   = (x - x_mean.unsqueeze(-1)) * mask_f
            # unbiased variance (divide by N-1) — matches original behaviour
            x_var  = (diff ** 2).sum(dim=-1) / (counts - 1).clamp(min=1)  # [B, C]
        else:
            x_mean = x.mean(dim=-1)                         # [B, C]
            x_var  = x.var(dim=-1, unbiased=True)           # [B, C]
        x_norm = (x - x_mean.unsqueeze(-1)) / (x_var.unsqueeze(-1).sqrt() + 1e-5)
        return x_norm, x_mean, x_var
    return x, None, None


def _get_seq_len(seq_len, n_fft, hop_length, stft_pad_amount):
    pad_amount = stft_pad_amount * 2 if stft_pad_amount is not None else n_fft // 2 * 2
    return ((seq_len + pad_amount - n_fft) // hop_length).clamp(min=0)


# ---------------------------------------------------------------------------
# AudioToMelSpectrogramPreprocessor (exact NeMo FilterbankFeatures replica)
# ---------------------------------------------------------------------------


class AudioToMelSpectrogramPreprocessor(nn.Module):
    """Exact replica of NeMo's AudioToMelSpectrogramPreprocessor → FilterbankFeatures.

    Uses librosa for mel filterbank (identical to NeMo) and torch.view_as_real
    for STFT magnitude computation (identical to NeMo).
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        n_mels=80,
        window_size=0.025,
        window_stride=0.01,
        dither=1e-5,
        pad_to=16,
        preemph=0.97,
        log_zero_guard_value=2**-24,
        mag_power=2.0,
        lowfreq=0.0,
        highfreq=None,
        frame_splicing=1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = int(window_size * sample_rate)  # 400
        self.hop_length = int(window_stride * sample_rate)  # 160
        self.n_fft = n_fft
        self.dither = dither
        self.pad_to = pad_to
        self.preemph = preemph
        self.log_zero_guard_value = log_zero_guard_value
        self.mag_power = mag_power
        self.frame_splicing = frame_splicing
        self.stft_pad_amount = None  # NeMo uses None for this model

        self.register_buffer(
            "window", torch.hann_window(self.win_length, periodic=False)
        )

        # Mel filterbank using librosa (exact match to NeMo's librosa.filters.mel)
        f_max = highfreq if highfreq is not None else sample_rate / 2
        mel_fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=lowfreq,
            fmax=f_max,
            norm="slaney",
        )
        self.register_buffer("fb", torch.tensor(mel_fb, dtype=torch.float).unsqueeze(0))

    def forward(self, input_signal, length=None):
        x = input_signal
        seq_len_time = (
            length
            if length is not None
            else torch.full(
                (x.shape[0],), x.shape[1], dtype=torch.long, device=x.device
            )
        )

        # Pad for exact STFT padding (None means no extra padding)
        if self.stft_pad_amount is not None:
            x = F.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        # Dither (only in training)
        if self.dither > 0 and self.training:
            x = x + self.dither * torch.randn_like(x)

        # Preemphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(
                0
            ) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0:1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        # STFT (center=True, pad_mode="constant" as in NeMo)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="constant",
        )

        # Magnitude: exact NeMo approach — view_as_real → sqrt(sum²)
        spec_real = torch.view_as_real(spec)
        mag = torch.sqrt(spec_real.pow(2).sum(-1))

        # Power spectrum
        if self.mag_power != 1.0:
            mag = mag.pow(self.mag_power)

        # Mel filterbank: fb is [1, n_mels, n_freqs], mag is [B, n_freqs, T]
        mel = torch.matmul(self.fb, mag)  # [B, n_mels, T]

        # Log
        mel = torch.log(mel + self.log_zero_guard_value)

        # Compute output sequence lengths
        seq_len_out = _get_seq_len(
            seq_len_time, self.n_fft, self.hop_length, self.stft_pad_amount
        )

        # Per-feature normalization (always applied)
        mel, _, _ = _normalize_batch(mel, seq_len_out, normalize_type="per_feature")

        # Mask beyond seq_len
        max_len = mel.shape[-1]
        mask = torch.arange(max_len, device=mel.device).repeat(
            mel.shape[0], 1
        ) >= seq_len_out.unsqueeze(1)
        mel = mel.masked_fill(mask.unsqueeze(1), 0.0)

        # Pad to multiple of pad_to
        if self.pad_to > 0:
            pad_amt = mel.shape[-1] % self.pad_to
            if pad_amt != 0:
                mel = F.pad(mel, (0, self.pad_to - pad_amt), value=0.0)

        return mel, seq_len_out


# ---------------------------------------------------------------------------
# TDNN / SE / ECAPA building blocks (exact NeMo replicas)
# ---------------------------------------------------------------------------


class TDNNModule(nn.Module):
    def __init__(
        self,
        inp_filters,
        out_filters,
        kernel_size=1,
        dilation=1,
        stride=1,
        padding=None,
    ):
        super().__init__()
        if padding is None:
            padding = _get_same_padding(kernel_size, stride, dilation)
        self.conv_layer = nn.Conv1d(
            inp_filters, out_filters, kernel_size, dilation=dilation, padding=padding
        )
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        x = self.conv_layer(x)
        x = self.activation(x)
        return self.bn(x)


class MaskedSEModule(nn.Module):
    def __init__(self, inp_filters, se_filters, out_filters, kernel_size=1, dilation=1):
        super().__init__()
        self.se_layer = nn.Sequential(
            nn.Conv1d(inp_filters, se_filters, kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(se_filters),
            nn.Conv1d(se_filters, out_filters, kernel_size, dilation=dilation),
            nn.Sigmoid(),
        )

    def forward(self, x, length=None):
        if length is None:
            s = x.mean(dim=2, keepdim=True)
        else:
            max_len = x.size(2)
            mask, num_values = _lens_to_mask(length, max_len=max_len, device=x.device)
            s = torch.sum((x * mask), dim=2, keepdim=True) / num_values
        return self.se_layer(s) * x


class TDNNSEModule(nn.Module):
    def __init__(
        self,
        inp_filters,
        out_filters,
        group_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        padding = _get_same_padding(kernel_size, dilation=dilation, stride=1)
        group_conv = nn.Conv1d(
            out_filters,
            out_filters,
            kernel_size,
            dilation=dilation,
            padding=padding,
            groups=group_scale,
        )
        self.group_tdnn_block = nn.Sequential(
            TDNNModule(inp_filters, out_filters, kernel_size=1, dilation=1),
            group_conv,
            nn.ReLU(),
            nn.BatchNorm1d(out_filters),
            TDNNModule(out_filters, out_filters, kernel_size=1, dilation=1),
        )
        self.se_layer = MaskedSEModule(out_filters, se_channels, out_filters)

    def forward(self, x, length=None):
        return self.se_layer(self.group_tdnn_block(x), length) + x


class ECAPAEncoder(nn.Module):
    def __init__(
        self,
        feat_in=80,
        filters=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 1, 1, 1, 1],
        scale=8,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            TDNNModule(
                feat_in, filters[0], kernel_size=kernel_sizes[0], dilation=dilations[0]
            )
        )
        for i in range(len(filters) - 2):
            self.layers.append(
                TDNNSEModule(
                    filters[i],
                    filters[i + 1],
                    group_scale=scale,
                    se_channels=128,
                    kernel_size=kernel_sizes[i + 1],
                    dilation=dilations[i + 1],
                )
            )
        self.feature_agg = TDNNModule(
            filters[-1],
            filters[-1],
            kernel_size=kernel_sizes[-1],
            dilation=dilations[-1],
        )

    def forward(self, audio_signal, length=None):
        x = audio_signal
        outputs = []
        for layer in self.layers:
            x = layer(x, length=length)
            outputs.append(x)
        x = torch.cat(outputs[1:], dim=1)
        x = self.feature_agg(x)
        return x, length


# ---------------------------------------------------------------------------
# Speaker Decoder (exact NeMo replica)
# ---------------------------------------------------------------------------


class AttentivePoolLayer(nn.Module):
    def __init__(self, inp_filters=3072, attention_channels=128):
        super().__init__()
        self.feat_in = 2 * inp_filters
        self.attention_layer = nn.Sequential(
            TDNNModule(inp_filters * 3, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, inp_filters, kernel_size=1),
        )

    def forward(self, x, length=None):
        max_len = x.size(2)
        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = _lens_to_mask(length, max_len=max_len, device=x.device)
        mean, std = _get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        attn = self.attention_layer(attn)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        alpha = F.softmax(attn, dim=2)
        mu, sg = _get_statistics_with_mask(x, alpha)

        return torch.cat((mu, sg), dim=1).unsqueeze(2)


class SpeakerDecoder(nn.Module):
    def __init__(
        self,
        feat_in=3072,
        num_classes=16681,
        emb_sizes=192,
        pool_mode="attention",
        attention_channels=128,
    ):
        super().__init__()
        self.angular = True  # Matches NeMo's AngularSoftmaxLoss
        self.emb_id = 2
        self._num_classes = num_classes
        self._pooling = AttentivePoolLayer(
            inp_filters=feat_in, attention_channels=attention_channels
        )

        emb_sizes = [emb_sizes] if isinstance(emb_sizes, int) else emb_sizes
        shapes = [self._pooling.feat_in]
        for s in emb_sizes:
            shapes.append(int(s))

        emb_layers = []
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = nn.Sequential(
                nn.BatchNorm1d(shape_in, affine=True, track_running_stats=True),
                nn.Conv1d(shape_in, shape_out, kernel_size=1),
            )
            emb_layers.append(layer)
        self.emb_layers = nn.ModuleList(emb_layers)
        bias = False if self.angular else True
        self.final = nn.Linear(shapes[-1], self._num_classes, bias=bias)

    def forward(self, encoder_output, length=None):
        pool = self._pooling(encoder_output, length)

        last_pool = pool
        for layer in self.emb_layers:
            last_pool = pool
            pool = layer(pool)
        emb = layer[: self.emb_id](last_pool)

        pool = pool.squeeze(-1)
        if self.angular:
            pool = F.normalize(pool, p=2, dim=1)
        out = self.final(pool)
        return out, emb.squeeze(-1)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class ECAPATDNN(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80, num_classes=16681, emb_size=192):
        super().__init__()
        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            n_fft=512,
            n_mels=n_mels,
            window_size=0.025,
            window_stride=0.01,
            dither=1e-5,
            pad_to=16,
            preemph=0.97,
            log_zero_guard_value=2**-24,
            mag_power=2.0,
        )
        self.encoder = ECAPAEncoder(
            feat_in=n_mels,
            filters=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 1, 1, 1, 1],
            scale=8,
        )
        self.decoder = SpeakerDecoder(
            feat_in=3072,
            num_classes=num_classes,
            emb_sizes=emb_size,
            pool_mode="attention",
            attention_channels=128,
        )

    def forward(self, audio):
        mel, mel_len = self.preprocessor(audio)
        encoded, enc_len = self.encoder(mel, mel_len)
        logits, embs = self.decoder(encoded, enc_len)
        return logits, embs


# ---------------------------------------------------------------------------
# Speaker Encoder Wrapper
# ---------------------------------------------------------------------------


class SpeakerEncoder:
    """Frozen ECAPA-TDNN speaker encoder."""

    def __init__(self, pt_path: str, device: torch.device, sample_rate: int = 16000):
        self.device = device
        self.sample_rate = sample_rate

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"ECAPA-TDNN checkpoint not found: {pt_path}")

        self.model = ECAPATDNN(sample_rate=sample_rate, n_mels=80, emb_size=192)
        self.model.eval()

        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        mapped_sd = self._map_state_dict(state_dict)
        missing, unexpected = self.model.load_state_dict(mapped_sd, strict=False)
        if missing:
            print(f"[SpeakerEncoder] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(
                f"[SpeakerEncoder] Unexpected keys ({len(unexpected)}): {unexpected[:5]}..."
            )

        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    def _map_state_dict(self, sd: dict) -> dict:
        mapped = {}
        for key, value in sd.items():
            new_key = key
            if key == "preprocessor.featurizer.window":
                new_key = "preprocessor.window"
            elif key == "preprocessor.featurizer.fb":
                new_key = "preprocessor.fb"
            mapped[new_key] = value
        return mapped

    @torch.no_grad()
    def get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        # audio is already on the training device — avoid redundant .to() transfer
        _, embs = self.model(audio)
        return embs


def speaker_loss(gen_emb: torch.Tensor, ref_emb: torch.Tensor) -> torch.Tensor:
    sim = F.cosine_similarity(gen_emb, ref_emb).mean()
    return 1.0 - sim
