"""Voice conversion analysis — speaker embedding comparison.

Compares ECAPA-TDNN embeddings of:
  - Profile (target voice)
  - Input audio (source)
  - Output audio (converted)

Returns cosine similarities and a human-readable quality summary.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("rvc_web.analysis")

# ---------------------------------------------------------------------------
# Singleton ECAPA encoder — loaded once per process
# ---------------------------------------------------------------------------

_encoder = None
_encoder_device = None


def _get_encoder(device: str = "cpu"):
    """Return a cached ECAPA-TDNN SpeakerEncoder instance."""
    global _encoder, _encoder_device
    if _encoder is not None and _encoder_device == device:
        return _encoder

    # Add backend/rvc to sys.path so 'infer' package resolves
    _pkg_dir = os.path.join(os.path.dirname(__file__), "..", "rvc")
    if _pkg_dir not in __import__("sys").path:
        __import__("sys").path.insert(0, _pkg_dir)

    from infer.lib.train.speaker_encoder import SpeakerEncoder

    project_root = os.environ.get("PROJECT_ROOT", "")
    if not project_root:
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        )

    pt_path = os.path.join(project_root, "assets", "ecapa", "ecapa_tdnn.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"ECAPA-TDNN checkpoint not found: {pt_path}")

    _encoder = SpeakerEncoder(pt_path=pt_path, device=torch.device(device))
    _encoder_device = device
    return _encoder


# ---------------------------------------------------------------------------
# Audio loading helper
# ---------------------------------------------------------------------------


def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and resample to target_sr (mono, float32)."""
    import librosa

    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y


def _extract_embedding(audio: np.ndarray, sr: int, encoder) -> np.ndarray:
    """Extract a single ECAPA-TDNN speaker embedding from audio of any length.

    Long audio is chunked and averaged rather than truncated, so a 10–12 minute
    file produces a stable, representative embedding:

      Chunking strategy
      -----------------
      chunk_duration : 3 s   — matches ECAPA-TDNN training segment length
      stride_duration: 2 s   — 1 s overlap; no speech near boundaries is lost
      A 10-min file  → ~299 chunks
      A 12-min file  → ~359 chunks

      Per-chunk processing
      --------------------
      Each 3 s chunk → ECAPA-TDNN encoder → 192-dim embedding vector.
      Short audio (< 3 s) is zero-padded to exactly 3 s before encoding.

      Aggregation
      -----------
      All chunk embeddings are stacked [N, 192] and averaged along dim 0,
      producing a single 192-dim vector that represents the "mean speaker
      identity" across the full file.  The result is L2-normalised to unit
      sphere (||emb||₂ = 1) so cosine similarity == dot product.

      Caveat: chunks are weighted equally — background noise or non-speech
      segments are included in the average.  For clean speech this is fine;
      for noisy recordings, VAD pre-filtering would improve accuracy.
    """
    chunk_duration = 3.0
    stride_duration = 2.0
    chunk_samples = int(chunk_duration * sr)
    stride_samples = int(stride_duration * sr)

    if len(audio) < chunk_samples:
        # Pad short audio
        audio = np.pad(audio, (0, chunk_samples - len(audio)), mode="constant")
        chunks = [audio]
    else:
        chunks = []
        start = 0
        while start + chunk_samples <= len(audio):
            chunks.append(audio[start : start + chunk_samples].copy())
            start += stride_samples
        # Final chunk
        if start < len(audio):
            final = audio[start:].copy()
            if len(final) < chunk_samples:
                final = np.pad(final, (0, chunk_samples - len(final)), mode="constant")
            chunks.append(final)

    embeddings = []
    for chunk in chunks:
        # Move audio to the encoder's device — numpy arrays are always CPU,
        # so we must explicitly transfer before passing to get_embedding.
        # (get_embedding no longer does .to(self.device) internally)
        audio_tensor = torch.from_numpy(chunk).unsqueeze(0).to(encoder.device)  # [1, T]
        with torch.no_grad():
            emb = encoder.get_embedding(audio_tensor)  # [1, 192]
        embeddings.append(emb)

    # Average and normalize
    all_emb = torch.cat(embeddings, dim=0)  # [N, 192]
    avg_emb = all_emb.mean(dim=0)  # [192]
    avg_emb = torch.nn.functional.normalize(avg_emb, p=2, dim=0)
    return avg_emb.cpu().numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quality_label(score: float) -> str:
    """Return a quality label for a similarity score [0, 1].

    Thresholds calibrated for ECAPA-TDNN cosine similarity on voice conversion
    output. Industry EER operating point on VoxCeleb1 sits around 0.75–0.80
    for same-speaker pairs; conversion adds deviation so thresholds are shifted
    slightly lower than pure speaker-verification benchmarks.

      >= 0.80  Excellent  — strong speaker match, conversion working well
      >= 0.70  Very Good  — clear similarity, minor voice character loss
      >= 0.60  Good       — recognisable match, some divergence
      >= 0.50  Moderate   — partial similarity, audible difference
      >= 0.40  Poor       — low match, significant voice character loss
      <  0.40  Very Poor  — near-random similarity, conversion largely failed
    """
    pct = score * 100
    if pct >= 80:
        return "Excellent"
    elif pct >= 70:
        return "Very Good"
    elif pct >= 60:
        return "Good"
    elif pct >= 50:
        return "Moderate"
    elif pct >= 40:
        return "Poor"
    else:
        return "Very Poor"


def analyze_conversion(
    profile_emb_path: str,
    input_audio_path: str,
    output_audio_path: str,
    device: str = "cpu",
) -> dict:
    """Analyze voice conversion quality via speaker embedding comparison.

    Args:
        profile_emb_path: Path to profile_embedding.pt (from training) or
                          directory with WAV files to compute from.
        input_audio_path: Path to the source audio file.
        output_audio_path: Path to the converted audio file.
        device: "cpu" or "mps" or "cuda".

    Returns:
        dict with similarity scores, quality labels, and summary text.
    """
    encoder = _get_encoder(device)

    # 1. Load profile embedding (or compute from audio)
    if os.path.isfile(profile_emb_path) and profile_emb_path.endswith(".pt"):
        prof_data = torch.load(profile_emb_path, map_location="cpu", weights_only=False)
        profile_emb = prof_data["profile_embedding"]  # [192]
        if isinstance(profile_emb, torch.Tensor):
            profile_emb = profile_emb.cpu().numpy()
    else:
        # Compute from audio directory — check both root and audio/ subdirectory
        import glob as _glob

        wav_files = sorted(_glob.glob(os.path.join(profile_emb_path, "*.wav")))
        if not wav_files:
            # Try audio/ subdirectory
            audio_subdir = os.path.join(profile_emb_path, "audio")
            wav_files = sorted(_glob.glob(os.path.join(audio_subdir, "*.wav")))
        if not wav_files:
            raise FileNotFoundError(
                f"No WAV files in {profile_emb_path} or audio/ subdirectory"
            )

        embs = []
        for wav_path in wav_files:
            audio = _load_audio(wav_path)
            emb = _extract_embedding(audio, 16000, encoder)
            embs.append(emb)
        profile_emb = np.mean(embs, axis=0)
        profile_emb = profile_emb / (np.linalg.norm(profile_emb) + 1e-10)

    # 2. Extract input and output embeddings
    input_audio = _load_audio(input_audio_path)
    output_audio = _load_audio(output_audio_path)

    input_emb = _extract_embedding(input_audio, 16000, encoder)
    output_emb = _extract_embedding(output_audio, 16000, encoder)

    # 3. Compute cosine similarities
    sim_profile_input = float(np.dot(profile_emb, input_emb))
    sim_profile_output = float(np.dot(profile_emb, output_emb))
    sim_input_output = float(np.dot(input_emb, output_emb))

    # Clamp to [0, 1] for display (cosine sim can be slightly outside due to float)
    sim_profile_input = np.clip(sim_profile_input, 0.0, 1.0)
    sim_profile_output = np.clip(sim_profile_output, 0.0, 1.0)
    sim_input_output = np.clip(sim_input_output, 0.0, 1.0)

    # 4. Compute improvement
    improvement = sim_profile_output - sim_profile_input
    improvement_pct = (
        (improvement / sim_profile_input * 100) if sim_profile_input > 0.01 else 0
    )

    # 5. Quality labels
    quality_output = quality_label(sim_profile_output)
    quality_input = quality_label(sim_profile_input)

    # 6. Summary text
    if improvement > 0.1:
        verdict = (
            f"✅ Excellent voice conversion! The output voice is {sim_profile_output:.0%} "
            f"similar to the target profile, up from {sim_profile_input:.0%} in the input. "
            f"This is a {improvement_pct:.0f}% improvement in speaker similarity."
        )
    elif improvement > 0.0:
        verdict = (
            f"👍 Good voice conversion. The output voice is {sim_profile_output:.0%} "
            f"similar to the target profile, slightly better than the input's "
            f"{sim_profile_input:.0%} similarity."
        )
    elif improvement > -0.1:
        verdict = (
            f"⚠️ Moderate conversion. The output voice is {sim_profile_output:.0%} "
            f"similar to the target profile, close to the input's "
            f"{sim_profile_input:.0%} similarity. The voice character hasn't "
            f"changed significantly."
        )
    else:
        verdict = (
            f"❌ Poor conversion. The output voice ({sim_profile_output:.0%} similarity) "
            f"is actually less similar to the target than the input "
            f"({sim_profile_input:.0%}). Try adjusting index_rate or retraining."
        )

    return {
        "profile_input_similarity": round(sim_profile_input, 4),
        "profile_output_similarity": round(sim_profile_output, 4),
        "input_output_similarity": round(sim_input_output, 4),
        "improvement": round(improvement, 4),
        "improvement_pct": round(improvement_pct, 1),
        "quality_input": quality_input,
        "quality_output": quality_output,
        "summary": verdict,
        # Full embedding vectors for radar chart (all 192 dimensions)
        "profile_emb": profile_emb.tolist(),
        "input_emb": input_emb.tolist(),
        "output_emb": output_emb.tolist(),
    }


# ---------------------------------------------------------------------------
# analyze_pair — generic two-file comparison (used by /api/analysis/compare)
# ---------------------------------------------------------------------------

def _audio_duration(path: str) -> float:
    """Return duration in seconds without loading the full file into RAM."""
    import soundfile as _sf
    info = _sf.info(path)
    return info.duration


def analyze_pair(
    path_a: str,
    path_b: str,
    profile_emb_path: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    """Compare two audio files using ECAPA-TDNN embeddings.

    path_a  — first audio file (e.g. source / input)
    path_b  — second audio file (e.g. converted / output)
    profile_emb_path — optional profile_embedding.pt or profile dir

    Returns a dict matching CompareResponse schema.
    """
    import os as _os

    encoder = _get_encoder(device)

    audio_a = _load_audio(path_a)
    audio_b = _load_audio(path_b)

    emb_a = _extract_embedding(audio_a, 16000, encoder)
    emb_b = _extract_embedding(audio_b, 16000, encoder)

    dur_a = _audio_duration(path_a)
    dur_b = _audio_duration(path_b)

    # a ↔ b similarity
    sim_ab = float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))

    # Profile comparisons (optional)
    emb_profile = None
    sim_pa = sim_pb = improvement = improvement_pct = None
    q_pa = q_pb = None

    if profile_emb_path:
        if _os.path.isfile(profile_emb_path) and profile_emb_path.endswith(".pt"):
            prof_data = torch.load(profile_emb_path, map_location="cpu", weights_only=False)
            profile_emb_t = prof_data["profile_embedding"]
            emb_profile = (
                profile_emb_t.cpu().numpy()
                if isinstance(profile_emb_t, torch.Tensor)
                else np.array(profile_emb_t)
            )
        else:
            import glob as _glob
            wav_files = sorted(_glob.glob(_os.path.join(profile_emb_path, "*.wav")))
            if not wav_files:
                wav_files = sorted(_glob.glob(_os.path.join(profile_emb_path, "audio", "*.wav")))
            if wav_files:
                embs = [_extract_embedding(_load_audio(p), 16000, encoder) for p in wav_files]
                emb_profile = np.mean(embs, axis=0)
                emb_profile /= np.linalg.norm(emb_profile) + 1e-10

        if emb_profile is not None:
            sim_pa = float(np.clip(np.dot(emb_profile, emb_a), 0.0, 1.0))
            sim_pb = float(np.clip(np.dot(emb_profile, emb_b), 0.0, 1.0))
            improvement = round(sim_pb - sim_pa, 4)
            improvement_pct = round(
                (improvement / sim_pa * 100) if sim_pa > 0.01 else 0.0, 1
            )
            q_pa = quality_label(sim_pa)
            q_pb = quality_label(sim_pb)

    # Summary
    if emb_profile is not None and sim_pa is not None and sim_pb is not None:
        if improvement > 0.1:
            summary = (
                f"✅ Strong conversion — output is {sim_pb:.0%} similar to target "
                f"(up from {sim_pa:.0%}, +{improvement_pct:.0f}%)."
            )
        elif improvement > 0.0:
            summary = (
                f"👍 Good conversion — output is {sim_pb:.0%} similar to target "
                f"(source was {sim_pa:.0%})."
            )
        elif improvement > -0.1:
            summary = (
                f"⚠️ Moderate conversion — output {sim_pb:.0%} vs source {sim_pa:.0%}. "
                f"Voice character hasn't changed significantly."
            )
        else:
            summary = (
                f"❌ Weak conversion — output ({sim_pb:.0%}) is less similar to target "
                f"than source ({sim_pa:.0%}). Try adjusting index_rate or retraining."
            )
    else:
        pct = round(sim_ab * 100)
        if pct > 85:
            summary = f"🔊 Files are very similar speakers ({pct}% similarity)."
        elif pct > 65:
            summary = f"🔊 Files share moderate speaker characteristics ({pct}% similarity)."
        else:
            summary = f"🔊 Files are from different speakers ({pct}% similarity)."

    return {
        "ab_similarity": round(sim_ab, 4),
        "profile_a_similarity": round(sim_pa, 4) if sim_pa is not None else None,
        "profile_b_similarity": round(sim_pb, 4) if sim_pb is not None else None,
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "quality_a": quality_label(sim_ab),   # quality of a↔b match
        "quality_b": quality_label(sim_ab),
        "quality_profile_a": q_pa,
        "quality_profile_b": q_pb,
        "summary": summary,
        "emb_a": emb_a.tolist(),
        "emb_b": emb_b.tolist(),
        "emb_profile": emb_profile.tolist() if emb_profile is not None else None,
        "duration_a": round(dur_a, 2),
        "duration_b": round(dur_b, 2),
    }
