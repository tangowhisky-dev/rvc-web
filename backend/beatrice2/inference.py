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
        import os as _os
        self._device = torch.device(device)
        # Store project root for f0_transform import inside the subprocess
        self._rvc_root = _os.environ.get("PROJECT_ROOT", _os.getcwd())
        # Persistent velocity state for realtime (reset when f0_norm_params changes)
        self._vel_state_rt = None
        self._last_norm_key: str = ""
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
        vq_topk = h.get("vq_topk", 4)
        phone_noise_ratio = h.get("phone_noise_ratio", 0.5)
        floor_noise_level = h.get("floor_noise_level", 1e-3)
        use_reference_encoder = bool(h.get("use_reference_encoder", False))
        reference_encoder_channels = int(h.get("reference_encoder_channels", 256))

        self.phone_extractor = PhoneExtractor()
        self.pitch_estimator = PitchEstimator()

        # ConverterNetwork: pass all architectural hyperparams from h so the
        # model matches the trained configuration exactly.
        self.net_g = ConverterNetwork(
            phone_extractor=self.phone_extractor,
            pitch_estimator=self.pitch_estimator,
            n_speakers=n_speakers,
            pitch_bins=pitch_bins,
            hidden_channels=hidden_channels,
            vq_topk=vq_topk,
            training_time_vq="none",
            phone_noise_ratio=phone_noise_ratio,
            floor_noise_level=floor_noise_level,
            use_reference_encoder=use_reference_encoder,
            reference_encoder_channels=reference_encoder_channels,
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

        # Install VQ hook so phone_extractor.head output is routed through
        # the VQ codebooks at inference time (same as during training eval).
        self.net_g.enable_hook()

        # Cache hyperparams
        self._in_sample_rate: int = checkpoint.get("in_sample_rate", self.IN_SR)
        self._out_sample_rate: int = checkpoint.get("out_sample_rate", self.OUT_SR)
        self._use_reference_encoder: bool = use_reference_encoder

        # Load mean style vector (default, used when no reference clip is given)
        self._mean_style_vec: "torch.Tensor | None" = None
        checkpoint_dir = os.path.dirname(checkpoint_path)
        mean_style_path = os.path.join(checkpoint_dir, "style_vec_mean.pt")
        if os.path.isfile(mean_style_path):
            import torch as _torch
            self._mean_style_vec = _torch.load(
                mean_style_path, map_location=self._device, weights_only=True
            )

        # Session-level cached style vector (set by set_reference())
        self._session_style_vec: "torch.Tensor | None" = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # F0 normalization hook
    # ------------------------------------------------------------------

    def _apply_f0_norm_patch(self, f0_norm_params: dict,
                               vel_state=None) -> None:
        """Monkey-patch pitch_estimator.sample_pitch to apply the full F0 prior pipeline.

        Transform order (all in bin space — bins are linear in log-Hz):
          1. Affine: bin_out = μ_t_bin + (σ_t/σ_s) × (bin_in − μ_s_bin)
          2. Velocity normalization (if vel_ratio present; causal, uses vel_state)
          3. Soft-clip to target speaking range (if p5_tgt/p95_tgt present)

        Unvoiced frames (bin = 0) are left unchanged throughout.
        Called once before inference; _remove_f0_norm_patch restores the original.

        Parameters
        ----------
        vel_state : VelocityNormalizerBins | None
            Persistent velocity state for realtime (shared across blocks).
            Pass None for offline (a fresh one-shot state is created per call).
        """
        import math
        import torch
        import sys, os as _os
        # Ensure project root is importable
        _proj = getattr(self, '_rvc_root', None) or _os.getcwd()
        if _proj not in sys.path:
            sys.path.insert(0, _proj)
        from backend.app.f0_transform import apply_f0_prior_bins, VelocityNormalizerBins

        bpo = self.pitch_estimator.pitch_bins_per_octave  # 96
        src_mean = float(f0_norm_params["src_mean"])
        src_std  = float(f0_norm_params["src_std"])
        tgt_mean = float(f0_norm_params["tgt_mean"])
        tgt_std  = float(f0_norm_params["tgt_std"])

        mu_s_bin = math.log2(src_mean / 55.0) * bpo
        mu_t_bin = math.log2(tgt_mean / 55.0) * bpo
        ratio = (tgt_std / src_std) if src_std > 1e-6 else 1.0

        # Velocity state: use caller-provided (realtime) or create fresh (offline)
        _vel_state = vel_state
        vel_ratio = f0_norm_params.get("vel_ratio")
        if _vel_state is None and vel_ratio is not None and abs(float(vel_ratio) - 1.0) > 1e-4:
            _vel_state = VelocityNormalizerBins(float(vel_ratio))

        _f0_params = f0_norm_params  # captured
        _has_affine = src_mean > 0 and tgt_mean > 0

        _orig_sample_pitch = self.pitch_estimator.sample_pitch

        def _normed_sample_pitch(pitch_logits, band_width=4, return_features=False):
            result = _orig_sample_pitch(pitch_logits, band_width=band_width,
                                        return_features=return_features)
            qp = result[0] if return_features else result
            # Step 1: Affine in bin space (only when src/tgt means are available)
            if _has_affine:
                voiced = qp > 0
                if voiced.any():
                    qp_f = qp.float()
                    qp_norm = torch.round(
                        mu_t_bin + ratio * (qp_f - mu_s_bin)
                    ).long()
                    qp = torch.where(voiced, qp_norm.clamp(1, self.net_g.pitch_bins - 1), qp)
            # Steps 2-3: velocity + soft-clip via f0_transform (apply even without affine)
            qp = apply_f0_prior_bins(qp, _f0_params, vel_state=_vel_state)
            if return_features:
                return qp, result[1]
            return qp

        self.pitch_estimator._orig_sample_pitch = _orig_sample_pitch
        self.pitch_estimator.sample_pitch = _normed_sample_pitch

    def _remove_f0_norm_patch(self) -> None:
        """Restore the original sample_pitch after a normalized inference run."""
        orig = getattr(self.pitch_estimator, "_orig_sample_pitch", None)
        if orig is not None:
            self.pitch_estimator.sample_pitch = orig
            del self.pitch_estimator._orig_sample_pitch

    # ------------------------------------------------------------------
    # Reference encoder API
    # ------------------------------------------------------------------

    @property
    def has_reference_encoder(self) -> bool:
        """True when the loaded checkpoint was trained with use_reference_encoder."""
        return getattr(self, "_use_reference_encoder", False)

    def encode_reference(self, wav_16k: np.ndarray) -> "np.ndarray":
        """Encode a reference waveform into a style vector.

        Parameters
        ----------
        wav_16k : np.ndarray
            1-D float32 waveform at 16 000 Hz.  5–10 s recommended.

        Returns
        -------
        np.ndarray
            1-D float32 style vector of shape ``(style_channels,)``.
            Returns zeros if the model has no reference encoder.
        """
        import torch
        if not self.has_reference_encoder:
            style_channels = self.net_g.reference_encoder.proj.out_features
            return np.zeros(style_channels, dtype=np.float32)
        wav = torch.from_numpy(wav_16k).float().to(self._device)
        with torch.inference_mode():
            style_vec = self.net_g.reference_encoder(wav)  # [style_channels]
        return style_vec.cpu().numpy().astype(np.float32)

    def set_reference(self, wav_16k: np.ndarray) -> None:
        """Encode ``wav_16k`` and cache the style vector for all subsequent calls.

        Called once at realtime session start.  Cached until explicitly replaced
        or cleared.  Pass ``wav_16k=None`` to revert to the profile mean style.
        """
        import torch
        if wav_16k is None:
            self._session_style_vec = None
            return
        style_np = self.encode_reference(wav_16k)
        self._session_style_vec = torch.from_numpy(style_np).to(self._device)

    def _get_active_style_vec(self) -> "torch.Tensor | None":
        """Return the session style vector, falling back to the profile mean, then None."""
        if self._session_style_vec is not None:
            return self._session_style_vec
        if self._mean_style_vec is not None:
            return self._mean_style_vec
        return None

    def convert(
        self,
        audio_16k: np.ndarray,
        target_speaker_id: int = 0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_semitones: float = 0.0,
        f0_norm_params: "dict | None" = None,
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
        # When normalization is active, the sample_pitch patch handles all pitch
        # mapping — pass shift=0 so the scalar bin-offset in converter.py is a no-op.
        effective_shift = 0.0 if f0_norm_params else float(pitch_shift_semitones)
        pitch_shift = torch.tensor([effective_shift], device=device)
        formant_shift = torch.tensor([float(formant_shift_semitones)], device=device)

        if f0_norm_params:
            # Manage persistent velocity state: reset if norm params changed
            _norm_key = str(sorted((k, v) for k, v in f0_norm_params.items()
                                   if k in ("src_mean", "src_std", "tgt_mean", "tgt_std",
                                            "vel_ratio", "p5_tgt", "p95_tgt")))
            if _norm_key != self._last_norm_key:
                self._vel_state_rt = None
                self._last_norm_key = _norm_key
            # Lazily create velocity state for realtime
            if self._vel_state_rt is None:
                vel_ratio = f0_norm_params.get("vel_ratio")
                if vel_ratio is not None and abs(float(vel_ratio) - 1.0) > 1e-4:
                    from backend.app.f0_transform import VelocityNormalizerBins
                    self._vel_state_rt = VelocityNormalizerBins(float(vel_ratio))
            self._apply_f0_norm_patch(f0_norm_params, vel_state=self._vel_state_rt)

        # Resolve active style vector (session reference → profile mean → None)
        _active_style = self._get_active_style_vec()  # Tensor [style_channels] or None

        try:
            with torch.inference_mode():
                # Build ref_wav argument: unsqueeze to [1, T] for the batch forward
                _ref_wav = _active_style  # not a raw wav — inject via pre-encoded style
                # We bypass the encoder at inference: directly set the encoded style
                # vector into net_g before calling forward, then clear it after.
                # This avoids recomputing the mel transform every block.
                if _active_style is not None and self.has_reference_encoder:
                    # Temporarily monkey-patch reference_encoder.forward to return
                    # the pre-computed vector (skip mel + conv + gru at inference).
                    _orig_ref_enc = self.net_g.reference_encoder.forward
                    _sv = _active_style
                    def _cached_ref_enc(wav, _sv=_sv):
                        return _sv.unsqueeze(0).expand(wav.shape[0], -1)
                    self.net_g.reference_encoder.forward = _cached_ref_enc
                    # net_g.forward will call reference_encoder(ref_wav) with a dummy wav
                    _dummy_ref = torch.zeros(wav.shape[0], 1, device=device)
                    converted = self.net_g(
                        wav.unsqueeze(1),
                        target_ids,
                        formant_shift,
                        pitch_shift,
                        ref_wav=_dummy_ref,
                    )  # [batch, T_out]
                    self.net_g.reference_encoder.forward = _orig_ref_enc
                else:
                    # net_g expects [batch, 1, T]
                    converted = self.net_g(
                        wav.unsqueeze(1),
                        target_ids,
                        formant_shift,
                        pitch_shift,
                    )  # [batch, T_out]
        finally:
            if f0_norm_params:
                self._remove_f0_norm_patch()

        # Trim to expected output length
        # Ratio: out_sr / in_sr * hop_length relationship
        # The vocoder decimates by 160/240 ratio implicitly; we trim proportionally
        expected_out = original_length * self._out_sample_rate // self._in_sample_rate
        out = converted.squeeze()  # [T_out] — remove both batch and channel dims
        if out.shape[-1] > expected_out:
            out = out[:expected_out]

        return out.cpu().numpy().astype(np.float32)

    def convert_offline(
        self,
        audio_16k: np.ndarray,
        target_speaker_id: int = 0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_semitones: float = 0.0,
        chunk_samples: int = 96000,
        f0_norm_params: "dict | None" = None,
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
        # When normalization is active, patch sample_pitch and pass shift=0.
        effective_shift = 0.0 if f0_norm_params else float(pitch_shift_semitones)
        pitch_shift = torch.tensor([effective_shift], device=device)
        formant_shift = torch.tensor([float(formant_shift_semitones)], device=device)

        if f0_norm_params:
            # Offline: fresh velocity state so it accumulates across all chunks sequentially
            _offline_vel = None
            vel_ratio = f0_norm_params.get("vel_ratio")
            if vel_ratio is not None and abs(float(vel_ratio) - 1.0) > 1e-4:
                from backend.app.f0_transform import VelocityNormalizerBins
                _offline_vel = VelocityNormalizerBins(float(vel_ratio))
            self._apply_f0_norm_patch(f0_norm_params, vel_state=_offline_vel)

        # Pre-compute reference encoder patch (avoids re-encoding mel every chunk)
        _active_style = self._get_active_style_vec()
        _orig_ref_enc = None
        if _active_style is not None and self.has_reference_encoder:
            _orig_ref_enc = self.net_g.reference_encoder.forward
            _sv = _active_style
            def _cached_ref_enc_offline(wav, _sv=_sv):
                return _sv.unsqueeze(0).expand(wav.shape[0], -1)
            self.net_g.reference_encoder.forward = _cached_ref_enc_offline

        try:
            chunks_out = []
            with torch.inference_mode():
                for start in range(0, T, chunk_samples):
                    chunk = wav[start : start + chunk_samples]  # [N]

                    # Pad last chunk to multiple of _HOP
                    rem = chunk.shape[0] % _HOP
                    if rem != 0:
                        chunk = F.pad(chunk, (0, _HOP - rem))

                    _chunk_batch = chunk.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
                    if _active_style is not None and self.has_reference_encoder:
                        _dummy_ref = torch.zeros(1, 1, device=device)
                        out = self.net_g(
                            _chunk_batch,
                            target_speaker_id=target_ids,
                            formant_shift_semitone=formant_shift,
                            pitch_shift_semitone=pitch_shift,
                            ref_wav=_dummy_ref,
                        )
                    else:
                        out = self.net_g(
                            _chunk_batch,
                            target_speaker_id=target_ids,
                            formant_shift_semitone=formant_shift,
                            pitch_shift_semitone=pitch_shift,
                        )
                    chunks_out.append(out.squeeze().cpu().float())
        finally:
            if f0_norm_params:
                self._remove_f0_norm_patch()
            if _orig_ref_enc is not None:
                self.net_g.reference_encoder.forward = _orig_ref_enc

        y_hat = torch.cat(chunks_out, dim=0)

        # Trim to exact expected output length (trim once after concat to avoid
        # per-chunk rounding drift)
        expected_samples = int(T * out_ratio)
        y_hat = y_hat[:expected_samples]

        # Peak normalization to prevent clipping (matches convert.py)
        peak = y_hat.abs().max()
        if peak > 0.95:
            y_hat = y_hat * (0.95 / peak)

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
