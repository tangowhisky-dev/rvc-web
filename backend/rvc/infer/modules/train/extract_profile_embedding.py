"""Extract a single robust speaker embedding from all profile audio.

Chunks all audio files into 2-4s segments, extracts ECAPA-TDNN embeddings
for each chunk, and averages them into one profile-level embedding.

This embedding is computed once before training and used as the fixed
speaker identity target throughout training.

Usage:
    python extract_profile_embedding.py /path/to/profile/audio/dir /path/to/output/embedding.pt
"""

import os
import sys
import glob
import torch
import numpy as np


def chunk_audio(audio, sample_rate, chunk_duration=3.0, stride_duration=2.0):
    """Split audio into overlapping chunks.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Sample rate of the audio.
        chunk_duration: Duration of each chunk in seconds.
        stride_duration: Stride between chunks in seconds.

    Returns:
        List of 1D numpy arrays (chunks).
    """
    chunk_samples = int(chunk_duration * sample_rate)
    stride_samples = int(stride_duration * sample_rate)

    if len(audio) < chunk_samples:
        # Pad short audio to chunk size
        audio = np.pad(audio, (0, chunk_samples - len(audio)), mode="constant")
        return [audio]

    chunks = []
    start = 0
    while start + chunk_samples <= len(audio):
        chunks.append(audio[start : start + chunk_samples].copy())
        start += stride_samples

    # Final chunk (may be shorter, pad if needed)
    if start < len(audio):
        final = audio[start:].copy()
        if len(final) < chunk_samples:
            final = np.pad(final, (0, chunk_samples - len(final)), mode="constant")
        chunks.append(final)

    return chunks


def extract_profile_embedding(
    audio_dir, pt_path, device="cpu", chunk_duration=3.0, stride_duration=2.0
):
    """Extract a single robust speaker embedding from all audio in a directory."""
    # Add backend/rvc to sys.path so 'infer' package is importable
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _rvc_root = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    if _rvc_root not in sys.path:
        sys.path.insert(0, _rvc_root)
    print(f"[ProfileEmb] Added to sys.path: {_rvc_root}")

    from infer.lib.train.speaker_encoder import SpeakerEncoder

    encoder = SpeakerEncoder(pt_path=pt_path, device=torch.device(device))

    # Find all WAV files
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")

    print(f"[ProfileEmbedding] Found {len(wav_files)} WAV files")

    all_embeddings = []
    total_chunks = 0

    for wav_path in wav_files:
        # Load audio using scipy (no librosa dependency)
        from scipy.io import wavfile

        sr, audio = wavfile.read(wav_path)

        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.float32:
            pass
        else:
            audio = audio.astype(np.float32)

        # Handle stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample

            new_len = int(len(audio) * 16000 / sr)
            audio = resample(audio, new_len)
            sr = 16000

        # Chunk the audio
        chunks = chunk_audio(audio, sr, chunk_duration, stride_duration)

        # Extract embeddings for each chunk
        for chunk in chunks:
            audio_tensor = torch.from_numpy(chunk).unsqueeze(0)  # [1, T]
            with torch.no_grad():
                emb = encoder.get_embedding(audio_tensor)  # [1, 192]
            all_embeddings.append(emb)
            total_chunks += 1

    if not all_embeddings:
        raise RuntimeError("No embeddings extracted — check audio files")

    # Average all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 192]
    profile_emb = all_embeddings.mean(dim=0)  # [192]

    # L2 normalize (ECAPA embeddings are typically L2 normalized)
    profile_emb = torch.nn.functional.normalize(profile_emb, p=2, dim=0)

    print(
        f"[ProfileEmbedding] Extracted {total_chunks} chunks from {len(wav_files)} files"
    )
    print(f"[ProfileEmbedding] Profile embedding shape: {profile_emb.shape}")
    print(f"[ProfileEmbedding] Embedding norm: {profile_emb.norm().item():.4f}")

    return profile_emb


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python extract_profile_embedding.py <audio_dir> <output_path> [device]"
        )
        sys.exit(1)

    audio_dir = sys.argv[1]
    output_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    print(f"[ProfileEmb] Starting extraction")
    print(f"[ProfileEmb] Audio dir: {audio_dir}")
    print(f"[ProfileEmb] Output: {output_path}")
    print(f"[ProfileEmb] Device: {device}")

    # Find the ECAPA-TDNN checkpoint
    project_root = os.environ.get("PROJECT_ROOT", "")
    print(f"[ProfileEmb] PROJECT_ROOT env: {project_root}")
    if not project_root:
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        )
        print(f"[ProfileEmb] PROJECT_ROOT fallback: {project_root}")
    pt_path = os.path.join(project_root, "assets", "ecapa", "ecapa_tdnn.pt")
    print(f"[ProfileEmb] ECAPA checkpoint: {pt_path}")
    print(f"[ProfileEmb] Checkpoint exists: {os.path.exists(pt_path)}")

    if not os.path.exists(pt_path):
        print(f"[ProfileEmb] ERROR: Checkpoint not found at {pt_path}")
        sys.exit(1)

    try:
        emb = extract_profile_embedding(audio_dir, pt_path, device=device)
        torch.save({"profile_embedding": emb}, output_path)
        print(f"[ProfileEmb] SUCCESS: Saved to {output_path}")
        print(f"[ProfileEmb] Embedding norm: {emb.norm().item():.4f}")
    except Exception as e:
        print(f"[ProfileEmb] ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
