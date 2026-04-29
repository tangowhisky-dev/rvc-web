"""ReferenceEncoder — extracts a fixed-size style embedding from a mel spectrogram.

Architecture (Skerry-Ryan et al. 2018, "Towards End-to-End Prosody Transfer"):
  6 × Conv2d (striding down time and freq) → GRU (summarise time) → Linear projection

The encoder compresses the *suprasegmental* character of a reference clip
(tempo, energy, pitch range, speaking style) into a ``style_channels``-dim
vector.  During training it receives a *different clip from the same speaker*
as the training sample, which forces it to learn speaker style rather than
content (reference leakage prevention).

At inference the encoder is applied once to 5–10s of reference audio.  The
resulting ``style_vec`` is broadcast additively into every synthesis block
via ``ConverterNetwork.style_projection``.

The module is zero-initialised at construction so that a fine-tune from a
pre-trained checkpoint that does NOT have the encoder behaves identically to
the baseline for the first steps; the encoder contribution grows as training
progresses.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torchaudio


# Fixed mel configuration — must match what the encoder sees at inference.
_MEL_SAMPLE_RATE    = 16_000   # input is 16 kHz (same as PhoneExtractor)
_MEL_N_FFT          = 512
_MEL_WIN_LENGTH     = 512
_MEL_HOP_LENGTH     = 160      # 10 ms
_MEL_N_MELS         = 80
_MEL_FMIN           = 50.0
_MEL_FMAX           = 7600.0


class ReferenceEncoder(nn.Module):
    """Maps a mel spectrogram to a style embedding vector.

    Parameters
    ----------
    style_channels : int
        Output dimension.  Default 256 matches ``hidden_channels`` in
        ``ConverterNetwork`` so the projection is a cheap identity-scale op.
    gru_hidden : int
        GRU hidden size.  Widened to 128 by default; 256 with more data.
    """

    def __init__(
        self,
        style_channels: int = 256,
        gru_hidden: int = 128,
    ) -> None:
        super().__init__()

        # 6-layer Conv2D pyramid.  After 6 strides of 2 along both axes:
        #   freq: 80 → 2  (80 // 2^6 = 1, padded to 2)
        #   time: T  → T//64   (used as sequence for GRU)
        channels = [1, 32, 32, 64, 64, 128, 128]
        convs = []
        for i in range(6):
            convs.append(
                nn.Conv2d(
                    channels[i], channels[i + 1],
                    kernel_size=3, stride=2, padding=1,
                )
            )
            convs.append(nn.BatchNorm2d(channels[i + 1]))
            convs.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*convs)

        # After conv stack: [B, 128, freq//64, T//64].  freq//64 of 80-mel = 2.
        gru_input_size = 128 * 2  # 128 channels × 2 freq bins

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(gru_hidden, style_channels)

        # Zero-initialise projection so the encoder starts as a no-op.
        # The model behaves like the baseline at the start of fine-tuning
        # and gradually learns to use the encoder as training progresses.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Pre-built mel transform (registered as non-parameter buffer carrier).
        # We register the transform as a module so it moves to device with .to().
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=_MEL_SAMPLE_RATE,
            n_fft=_MEL_N_FFT,
            win_length=_MEL_WIN_LENGTH,
            hop_length=_MEL_HOP_LENGTH,
            n_mels=_MEL_N_MELS,
            f_min=_MEL_FMIN,
            f_max=_MEL_FMAX,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )

    # ------------------------------------------------------------------
    # Audio → mel helper (shared between training and inference)
    # ------------------------------------------------------------------

    def wav_to_mel(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Convert a 16 kHz waveform to a log-mel spectrogram.

        Parameters
        ----------
        wav_16k : Tensor [B, T] or [T]
            16 kHz mono audio.  Batch dimension is optional.

        Returns
        -------
        Tensor [B, 1, n_mels, T_mel]
            Log-mel spectrogram, batch-first, with channel dim for Conv2d.
        """
        squeeze = wav_16k.dim() == 1
        if squeeze:
            wav_16k = wav_16k.unsqueeze(0)  # [1, T]
        mel = self.mel_transform(wav_16k)           # [B, n_mels, T_mel]
        mel = mel.clamp(min=1e-10).log_()           # log-mel
        mel = mel.unsqueeze(1)                      # [B, 1, n_mels, T_mel]
        if squeeze:
            mel = mel.squeeze(0)
        return mel

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Encode a reference waveform to a style vector.

        Parameters
        ----------
        wav_16k : Tensor [B, T] or [T]
            16 kHz mono audio.  Minimum ~0.5s recommended (80 mel frames).

        Returns
        -------
        Tensor [B, style_channels] or [style_channels]
            Style embedding.  Zero-projection output = zero vector at init.
        """
        squeeze = wav_16k.dim() == 1
        if squeeze:
            wav_16k = wav_16k.unsqueeze(0)

        mel = self.wav_to_mel(wav_16k)      # [B, 1, 80, T_mel]
        x = self.convs(mel)                 # [B, 128, ~2, T_mel//64]

        B, C, F, T_mel = x.shape
        # flatten freq into channel dim, make time the sequence axis
        x = x.permute(0, 3, 1, 2).reshape(B, T_mel, C * F)  # [B, T_mel//64, 128*F]

        _, h = self.gru(x)                  # h: [1, B, gru_hidden]
        h = h.squeeze(0)                    # [B, gru_hidden]
        out = self.proj(h)                  # [B, style_channels]

        if squeeze:
            out = out.squeeze(0)
        return out

    def encode_clip(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Alias of forward() for clarity at inference call sites."""
        return self.forward(wav_16k)
