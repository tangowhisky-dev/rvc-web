"""Voice conversion analysis — speaker embedding comparison.

Compares ECAPA-TDNN embeddings of:
  - Profile (target voice)
  - Input audio (source)
  - Output audio (converted)

Returns cosine similarities and a human-readable quality summary.
"""

import os
import logging

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
    _pkg_dir = os.path.join(os.path.dirname(__file__), "..", "..", "rvc")
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
    """Extract a single ECAPA embedding from audio.

    Chunks long audio into 3s segments with 2s stride, extracts embeddings
    for each chunk, and averages them (same method as training).
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
        audio_tensor = torch.from_numpy(chunk).unsqueeze(0)  # [1, T]
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
    """Return a quality label for a similarity score [0, 1]."""
    pct = score * 100
    if pct > 90:
        return "Excellent"
    elif pct > 80:
        return "Good"
    elif pct > 60:
        return "Normal"
    elif pct > 40:
        return "Bad"
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
        # Compute from audio directory
        import glob as _glob

        wav_files = sorted(_glob.glob(os.path.join(profile_emb_path, "*.wav")))
        if not wav_files:
            raise FileNotFoundError(f"No WAV files in {profile_emb_path}")

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
        # Embedding vectors for radar chart (first 10 dimensions)
        "profile_emb_top10": profile_emb[:10].tolist(),
        "input_emb_top10": input_emb[:10].tolist(),
        "output_emb_top10": output_emb[:10].tolist(),
    }
