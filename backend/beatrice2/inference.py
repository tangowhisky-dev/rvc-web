"""BeatriceInferenceEngine — loads a Beatrice 2 checkpoint and runs voice conversion.

Uses the Python (.pt.gz) checkpoint path, NOT the binary .bin format (which
is for the C++ runtime). CUDA-only as per the Beatrice 2 architecture constraint.

Output sample rate: 24 000 Hz.
Callers (offline router) are responsible for writing output at 24 kHz or
resampling to a target SR.
"""

from __future__ import annotations

import gzip
import os
from typing import Optional

import numpy as np


class BeatriceInferenceEngine:
    """Loads checkpoint_latest.pt.gz and runs voice conversion via ConverterNetwork.

    Parameters
    ----------
    checkpoint_path : str
        Path to ``checkpoint_latest.pt.gz`` produced by the Beatrice 2 trainer.
    device : str
        ``"cuda"`` (required — Beatrice 2 does not run on MPS/CPU in inference).
    """

    IN_SR: int = 16_000   # Input sample rate expected by PhoneExtractor
    OUT_SR: int = 24_000  # Output sample rate produced by Vocoder

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        import torch

        self._device = torch.device(device)
        self._load(checkpoint_path)

    # ------------------------------------------------------------------
    # Internal load
    # ------------------------------------------------------------------

    def _load(self, checkpoint_path: str) -> None:
        import torch
        from backend.beatrice2.beatrice_trainer.models import (
            PhoneExtractor,
            PitchEstimator,
            ConverterNetwork,
        )

        with gzip.open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)

        # Reconstruct models from checkpoint
        h = checkpoint.get("h", {})
        n_speakers = h.get("n_speakers", 1)
        pitch_bins = h.get("pitch_bins", 448)
        hidden_channels = h.get("hidden_channels", 256)

        self.phone_extractor = PhoneExtractor()
        self.pitch_estimator = PitchEstimator()

        # ConverterNetwork needs architectural hyperparams from h
        self.net_g = ConverterNetwork(
            phone_extractor=self.phone_extractor,
            pitch_estimator=self.pitch_estimator,
            n_speakers=n_speakers,
            pitch_bins=pitch_bins,
            hidden_channels=hidden_channels,
        )

        # Load weights
        if "phone_extractor" in checkpoint:
            self.phone_extractor.load_state_dict(
                checkpoint["phone_extractor"], strict=False
            )
        if "pitch_estimator" in checkpoint:
            self.pitch_estimator.load_state_dict(
                checkpoint["pitch_estimator"]
            )
        if "net_g" in checkpoint:
            self.net_g.load_state_dict(checkpoint["net_g"], strict=False)

        # Move to device and set eval
        self.phone_extractor = (
            self.phone_extractor.to(self._device).eval().requires_grad_(False)
        )
        self.pitch_estimator = (
            self.pitch_estimator.to(self._device).eval().requires_grad_(False)
        )
        self.net_g = self.net_g.to(self._device).eval().requires_grad_(False)

        # Cache hyperparams
        self._in_sample_rate: int = checkpoint.get("in_sample_rate", self.IN_SR)
        self._out_sample_rate: int = checkpoint.get("out_sample_rate", self.OUT_SR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        audio_16k: np.ndarray,
        target_speaker_id: int = 0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_semitones: float = 0.0,
    ) -> np.ndarray:
        """Convert ``audio_16k`` to the target speaker voice.

        Parameters
        ----------
        audio_16k : np.ndarray
            1-D float32 waveform at 16 000 Hz.
        target_speaker_id : int
            Speaker ID to convert to (0 for single-speaker profiles).
        pitch_shift_semitones : float
            Semitones to shift pitch (positive = up).
        formant_shift_semitones : float
            Semitones to shift formants (positive = up).

        Returns
        -------
        np.ndarray
            1-D float32 waveform at ``OUT_SR`` (24 000 Hz).
        """
        import torch
        import torch.nn.functional as F

        device = self._device

        # Convert to tensor [1, T]
        wav = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)

        # Resample if needed
        if self._in_sample_rate != self.IN_SR:
            import torchaudio
            wav = torchaudio.transforms.Resample(
                self._in_sample_rate, self.IN_SR
            ).to(device)(wav)

        original_length = wav.shape[-1]

        # Pad to multiple of in_sample_rate to reduce shape variety in cache
        pad_to = self.IN_SR
        if wav.shape[-1] % pad_to != 0:
            pad_len = pad_to - wav.shape[-1] % pad_to
            wav = F.pad(wav, (0, pad_len))

        target_ids = torch.tensor([target_speaker_id], device=device)
        pitch_shift = torch.tensor([float(pitch_shift_semitones)], device=device)
        formant_shift = torch.tensor([float(formant_shift_semitones)], device=device)

        with torch.inference_mode():
            # net_g expects [batch, 1, T]
            converted = self.net_g(
                wav.unsqueeze(1),
                target_ids,
                formant_shift,
                pitch_shift,
            )  # [batch, T_out]

        # Trim to expected output length
        # Ratio: out_sr / in_sr * hop_length relationship
        # The vocoder decimates by 160/240 ratio implicitly; we trim proportionally
        expected_out = original_length * self._out_sample_rate // self._in_sample_rate
        out = converted.squeeze()  # [T_out] — remove both batch and channel dims
        if out.shape[-1] > expected_out:
            out = out[:expected_out]

        return out.cpu().numpy().astype(np.float32)

    def convert_long(
        self,
        audio_16k: np.ndarray,
        target_speaker_id: int = 0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_semitones: float = 0.0,
        chunk_samples: int = 96000,
    ) -> np.ndarray:
        """Convert a long audio file by chunking into 6-second windows.

        Safe for files of any length — processes ``chunk_samples`` frames at a
        time (default 96000 = 6s @ 16kHz, matching the ``wav_length`` the model
        was trained on).  Chunks are concatenated and trimmed to the exact
        expected output length.  No crossfade — Beatrice 2 is fully convolutional
        with local attention; hard chunk boundaries produce at most a 10ms
        discontinuity which is inaudible in practice.

        Used by the offline inference router.  Realtime uses ``convert()``
        directly since it already manages its own short rolling window.

        Parameters
        ----------
        audio_16k : np.ndarray
            1-D float32 waveform at 16 000 Hz, any length.
        chunk_samples : int
            Chunk size in input samples.  Must be a multiple of 160 (hop
            length).  Default 96000 matches training ``wav_length``.

        Returns
        -------
        np.ndarray
            1-D float32 waveform at ``OUT_SR`` (24 000 Hz).
        """
        import torch
        import torch.nn.functional as F

        _HOP = 160  # phone extractor hop length — all chunks must be multiples of this
        assert chunk_samples % _HOP == 0, "chunk_samples must be a multiple of 160"

        device = self._device
        out_ratio = self._out_sample_rate / self._in_sample_rate  # 1.5

        wav = torch.from_numpy(audio_16k).float().to(device)  # [T]

        # Resample if model was saved with a different in_sample_rate
        if self._in_sample_rate != self.IN_SR:
            import torchaudio
            wav = torchaudio.transforms.Resample(
                self._in_sample_rate, self.IN_SR
            ).to(device)(wav.unsqueeze(0)).squeeze(0)

        T = wav.shape[0]

        target_ids  = torch.tensor([target_speaker_id], device=device)
        pitch_shift = torch.tensor([float(pitch_shift_semitones)], device=device)
        formant_shift = torch.tensor([float(formant_shift_semitones)], device=device)

        chunks_out = []
        with torch.inference_mode():
            for start in range(0, T, chunk_samples):
                chunk = wav[start : start + chunk_samples]  # [N]

                # Pad last chunk to multiple of _HOP
                rem = chunk.shape[0] % _HOP
                if rem != 0:
                    chunk = F.pad(chunk, (0, _HOP - rem))

                # net_g expects [batch, 1, T]
                out = self.net_g(
                    chunk.unsqueeze(0).unsqueeze(0),
                    target_ids,
                    formant_shift,
                    pitch_shift,
                )  # [1, 1, T_out] or [1, T_out]
                chunks_out.append(out.squeeze().cpu().float())

        y_hat = torch.cat(chunks_out, dim=0)

        # Trim to exact expected output length (trim once after concat to avoid
        # per-chunk rounding drift)
        expected_samples = int(T * out_ratio)
        y_hat = y_hat[:expected_samples]

        return y_hat.numpy().astype(np.float32)

    def unload(self) -> None:
        """Release GPU memory."""
        import torch

        del self.phone_extractor
        del self.pitch_estimator
        del self.net_g
        if self._device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Module-level engine cache (keyed by profile_id)
# ---------------------------------------------------------------------------

_beatrice_engines: dict[str, BeatriceInferenceEngine] = {}


def get_or_load_engine(
    profile_id: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> BeatriceInferenceEngine:
    """Return a cached BeatriceInferenceEngine, loading from disk if needed."""
    if profile_id not in _beatrice_engines:
        _beatrice_engines[profile_id] = BeatriceInferenceEngine(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    return _beatrice_engines[profile_id]


def unload_engine(profile_id: str) -> None:
    """Unload and remove the cached engine for profile_id."""
    engine = _beatrice_engines.pop(profile_id, None)
    if engine is not None:
        engine.unload()
