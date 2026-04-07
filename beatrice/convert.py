#!/usr/bin/env python3
"""
beatrice/convert.py — offline voice conversion using a trained Beatrice 2 model.

Usage:
    python3 beatrice/convert.py \
        --model  output/beatrice_alice/ \
        --input  source.wav \
        --output converted.wav \
        [--speaker 0] \
        [--pitch-shift 0] \
        [--formant-shift 0] \
        [--device auto|cpu|cuda|mps]

--model should be the output directory you passed to train.sh (the directory that
contains checkpoint_latest.pt.gz).

Input audio: any format supported by torchaudio/torchcodec — wav, flac, ogg, mp3,
aac, m4a, etc.  Any sample rate, mono or stereo.

Output: 24 kHz mono WAV (Beatrice's native output rate).

Speaker IDs (--speaker): 0-indexed, matching the alphabetical order of subdirectory
names you used during training.
"""

import argparse
import gzip
import importlib.util
import sys
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional


def _get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


def _load_trainer_module():
    """Import beatrice_trainer.__main__ from the sibling trainer/ directory."""
    trainer_main = Path(__file__).parent / "trainer" / "beatrice_trainer" / "__main__.py"
    if not trainer_main.exists():
        raise FileNotFoundError(
            f"Trainer not found at {trainer_main}\n"
            "Run: cd beatrice/trainer && git lfs pull"
        )
    spec = importlib.util.spec_from_file_location("beatrice_trainer.__main__", trainer_main)
    mod = importlib.util.module_from_spec(spec)
    # Don't re-exec if already loaded (e.g. if called from train.sh context)
    if "beatrice_trainer.__main__" not in sys.modules:
        sys.modules["beatrice_trainer.__main__"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["beatrice_trainer.__main__"]


def load_model(model_dir: Path, device: torch.device):
    """
    Load a trained ConverterNetwork from model_dir.
    Looks for checkpoint_latest.pt.gz (preferred) then checkpoint_best.pt.gz.
    Returns (net_g, h) where h is the hyperparameter AttrDict.
    """
    model_dir = Path(model_dir)

    ckpt_path = None
    for name in ("checkpoint_latest.pt.gz", "checkpoint_best.pt.gz"):
        candidate = model_dir / name
        if candidate.exists():
            ckpt_path = candidate
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {model_dir}\n"
            "Expected checkpoint_latest.pt.gz or checkpoint_best.pt.gz.\n"
            "This file is created automatically during training."
        )

    mod = _load_trainer_module()

    print(f"Loading checkpoint: {ckpt_path.name}")
    with gzip.open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)

    h = mod.AttrDict(ckpt["h"])

    # Determine n_speakers from the embed_speaker weight shape
    w = ckpt["net_g"].get("embed_speaker.weight")
    n_speakers = w.shape[0] if w is not None else 1
    print(f"  speakers: {n_speakers},  steps: {ckpt.get('iteration', '?')}")

    phone_extractor = mod.PhoneExtractor().eval().requires_grad_(False).to(device)
    phone_extractor.load_state_dict(ckpt["phone_extractor"], strict=False)

    pitch_estimator = mod.PitchEstimator().eval().requires_grad_(False).to(device)
    pitch_estimator.load_state_dict(ckpt["pitch_estimator"])

    net_g = mod.ConverterNetwork(
        phone_extractor,
        pitch_estimator,
        n_speakers,
        h.pitch_bins,
        h.hidden_channels,
        h.vq_topk,
        "none",          # no VQ noise at inference
        h.phone_noise_ratio,
        h.floor_noise_level,
    ).eval().requires_grad_(False).to(device)

    net_g.load_state_dict(ckpt["net_g"])
    return net_g, h


def convert(
    model_dir: Path,
    input_path: Path,
    output_path: Path,
    speaker_id: int = 0,
    pitch_shift_semitones: int = 0,
    formant_shift_semitones: int = 0,
    device: torch.device = torch.device("cpu"),
):
    net_g, h = load_model(model_dir, device)

    in_sr = h.in_sample_rate    # 16000
    out_sr = h.out_sample_rate  # 24000

    # Load input — torchaudio+torchcodec handles wav/flac/ogg/mp3/aac/m4a/…
    print(f"Loading input: {input_path}")
    wav, sr = torchaudio.load(str(input_path))

    # Stereo → mono
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    # Resample to model input rate (16 kHz)
    if sr != in_sr:
        wav = torchaudio.functional.resample(wav, sr, in_sr)

    wav = wav.to(device)  # [1, T]

    print(f"  {sr} Hz → {in_sr} Hz, {wav.shape[1] / in_sr:.2f}s")

    # Process in chunks to avoid OOM on long files.
    # Chunk size must be % 160 == 0 (feature extractor hop length).
    # 6s @ 16kHz = 96000 samples matches wav_length used in training.
    chunk_samples = 96000  # 6s @ 16kHz, % 160 == 0
    # Output length ratio: out_sr / in_sr = 24000 / 16000 = 1.5
    out_ratio = out_sr / in_sr

    t_id = torch.tensor([speaker_id], dtype=torch.long, device=device)
    f_shift = torch.tensor([float(formant_shift_semitones)], dtype=torch.float, device=device)
    p_shift = torch.tensor([float(pitch_shift_semitones)], dtype=torch.float, device=device)

    T = wav.shape[1]
    chunks = []
    with torch.inference_mode():
        for start in range(0, T, chunk_samples):
            chunk = wav[:, start:start + chunk_samples]  # [1, N]
            # Pad last chunk to multiple of 160
            rem = chunk.shape[1] % 160
            if rem != 0:
                chunk = torch.nn.functional.pad(chunk, (0, 160 - rem))
            out = net_g(
                chunk.unsqueeze(0),  # [1, 1, N]
                t_id,
                f_shift,
                pitch_shift_semitone=p_shift,
            )  # [1, 1, T_out] or [1, T_out]
            chunks.append(out.squeeze(0).squeeze(0).cpu().float())

    y_hat = torch.cat(chunks, dim=0)

    # Trim to expected output length
    expected_samples = int(T * out_ratio)
    y_hat = y_hat[:expected_samples]

    # Normalise to prevent clipping
    peak = y_hat.abs().max()
    if peak > 0.95:
        y_hat = y_hat * (0.95 / peak)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), y_hat.unsqueeze(0), out_sr)
    print(f"Saved: {output_path}  ({out_sr} Hz, {y_hat.shape[0] / out_sr:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Beatrice 2 offline voice conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", required=True, type=Path,
                        help="Training output dir (contains checkpoint_latest.pt.gz)")
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="Input audio file (wav, mp3, flac, m4a, ogg, …)")
    parser.add_argument("-o", "--output", required=True, type=Path,
                        help="Output wav path")
    parser.add_argument("-s", "--speaker", type=int, default=0,
                        help="Target speaker ID (0-indexed, alphabetical dir order)")
    parser.add_argument("-p", "--pitch-shift", type=int, default=0,
                        help="Pitch shift in semitones (e.g. -3 lower, +3 higher)")
    parser.add_argument("-f", "--formant-shift", type=int, default=0,
                        help="Formant shift in semitones")
    parser.add_argument("-d", "--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device")
    args = parser.parse_args()

    device = _get_device(args.device)
    print(f"Device: {device}")

    convert(
        model_dir=args.model,
        input_path=args.input,
        output_path=args.output,
        speaker_id=args.speaker,
        pitch_shift_semitones=args.pitch_shift,
        formant_shift_semitones=args.formant_shift,
        device=device,
    )


if __name__ == "__main__":
    main()
