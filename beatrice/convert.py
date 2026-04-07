#!/usr/bin/env python3
"""
beatrice/convert.py — offline voice conversion using a trained Beatrice 2 model.

Usage:
    python convert.py \
        --model  beatrice/trainer/output/paraphernalia_<name>_<step>/ \
        --input  source.wav \
        --output converted.wav \
        [--speaker 0] \
        [--pitch-shift 0] \
        [--formant-shift 0] \
        [--device cpu|cuda|mps]

The model dir is the paraphernalia_*/ directory output by beatrice_trainer.
Speaker IDs start at 0 (matches the subdirectory order passed to -d during training).

This script performs pure-PyTorch inference — no closed beatrice.lib required.
It loads all three model weights directly from the .bin files in the paraphernalia dir.
"""

import argparse
import gzip
import io
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio


# ---------------------------------------------------------------------------
# Tiny binary loader — reads the custom .bin format dumped by beatrice-trainer
# ---------------------------------------------------------------------------

class _BinReader:
    """Reads the flat binary format written by dump_params() in the trainer."""

    def __init__(self, path: Path):
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                self._data = f.read()
        else:
            self._data = path.read_bytes()
        self._pos = 0

    def read_fp16(self, *shape) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        nbytes = n * 2
        raw = self._data[self._pos: self._pos + nbytes]
        self._pos += nbytes
        t = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape)
        return t.float()  # promote to fp32 for arithmetic

    def done(self) -> bool:
        return self._pos >= len(self._data)


# ---------------------------------------------------------------------------
# Model classes (minimal inference-only subset of the trainer architecture)
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Conv1d):
    def __init__(self, in_c, out_c, ks, stride=1, dilation=1, groups=1, bias=True, delay=0):
        padding = (ks - 1) * dilation - delay
        self.trim = (ks - 1) * dilation - 2 * delay
        super().__init__(in_c, out_c, ks, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out = super().forward(x)
        return out[:, :, : -self.trim] if self.trim > 0 else out


def _ws_forward(layer, x):
    """Scaled Weight Standardisation forward pass for WSConv1d."""
    w = layer.weight
    var, mean = torch.var_mean(w, [1, 2], keepdim=True)
    scale = (var + 1e-4).rsqrt()
    w = (w - mean) * scale
    return (nn.functional.conv1d(x, w, layer.bias,
                                 layer.stride, layer.padding,
                                 layer.dilation, layer.groups)
            * layer.gain)


class _WSConv1d(_CausalConv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fan_in = self.in_channels * self.kernel_size[0] // self.groups
        self.weight.data.normal_(0.0, math.sqrt(1.0 / fan_in))
        if self.bias is not None:
            self.bias.data.zero_()
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1))

    def forward(self, x):
        return _ws_forward(self, x)


# ------ PhoneExtractor ------ #

class _PhoneExtractor(nn.Module):
    """
    Inference-only PhoneExtractor.  Weights are loaded from phone_extractor.bin.
    Input: [B, 1, T] at 16 kHz.  Output: [B, T/160, 128] soft phone units.
    """

    def __init__(self):
        super().__init__()
        # Encoder: 3 strided causal convs (stride 2,4,5 → total 40×) → 1 conv
        self.enc = nn.Sequential(
            _WSConv1d(1, 32, 7, stride=1, delay=3),
            nn.GELU(),
            _WSConv1d(32, 64, 4, stride=2, delay=1),
            nn.GELU(),
            _WSConv1d(64, 64, 8, stride=4, delay=2),
            nn.GELU(),
            _WSConv1d(64, 128, 5, stride=5, delay=2),  # → 16kHz/160 = 100 Hz
            nn.GELU(),
            _WSConv1d(128, 128, 1, stride=1, delay=0),
        )
        # Self-attention (rc.0)
        self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.ln = nn.LayerNorm(128)
        # Head projection (source of kNN codebook targets during training)
        self.head = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        h = self.enc(x)  # [B, 128, L]
        h = h.transpose(1, 2)  # [B, L, 128]
        a, _ = self.attn(h, h, h)
        h = self.ln(h + a)
        return self.head(h)  # [B, L, 128]


# ------ PitchEstimator ------ #

class _PitchEstimator(nn.Module):
    """
    Inference-only PitchEstimator.
    Input: [B, 1, T] at 16 kHz.  Output: pitch logits [B, 448, L], energy [B, 1, L].
    """

    def __init__(self, pitch_bins: int = 448):
        super().__init__()
        self.pitch_bins = pitch_bins
        self.enc = nn.Sequential(
            _CausalConv1d(1, 32, 7, stride=1, delay=3),
            nn.GELU(),
            _CausalConv1d(32, 64, 4, stride=2, delay=1),
            nn.GELU(),
            _CausalConv1d(64, 64, 8, stride=4, delay=2),
            nn.GELU(),
            _CausalConv1d(64, 128, 5, stride=5, delay=2),
            nn.GELU(),
            _CausalConv1d(128, 256, 1),
        )
        self.pitch_head = nn.Conv1d(256, pitch_bins, 1)
        self.energy_head = nn.Conv1d(256, 1, 1)

    def forward(self, x: torch.Tensor):
        h = self.enc(x)
        return self.pitch_head(h), self.energy_head(h)


# ---------------------------------------------------------------------------
# Full conversion: load checkpoint.pt.gz and use the full trainer models
# ---------------------------------------------------------------------------

def _get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def _load_checkpoint_model(ckpt_path: Path, device: torch.device):
    """
    Load a full .pt.gz checkpoint from the trainer and return the
    ConverterNetwork (which embeds PhoneExtractor and PitchEstimator).
    This is the highest-quality path — uses the full training architecture.
    """
    # The trainer __main__.py is a single-file monolith; we exec it to get
    # the class definitions, then load state dicts.
    trainer_main = Path(__file__).parent / "trainer" / "beatrice_trainer" / "__main__.py"
    if not trainer_main.exists():
        raise FileNotFoundError(f"Trainer not found at {trainer_main}")

    # Import as module
    import importlib.util
    spec = importlib.util.spec_from_file_location("beatrice_trainer.__main__", trainer_main)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["beatrice_trainer.__main__"] = mod
    spec.loader.exec_module(mod)

    print(f"Loading checkpoint: {ckpt_path}")
    with gzip.open(ckpt_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)

    h = mod.AttrDict(ckpt["h"])
    n_speakers = sum(1 for k in ckpt["net_g"] if k.startswith("embed_speaker.weight"))
    if n_speakers == 0:
        # Count from embed_speaker weight shape
        w = ckpt["net_g"].get("embed_speaker.weight")
        n_speakers = w.shape[0] if w is not None else 1

    phone_extractor = mod.PhoneExtractor().eval().requires_grad_(False)
    phone_extractor.load_state_dict(ckpt["phone_extractor"], strict=False)

    pitch_estimator = mod.PitchEstimator().eval().requires_grad_(False)
    pitch_estimator.load_state_dict(ckpt["pitch_estimator"])

    net_g = mod.ConverterNetwork(
        phone_extractor,
        pitch_estimator,
        n_speakers,
        h.pitch_bins,
        h.hidden_channels,
        h.vq_topk,
        "none",  # no VQ during inference
        h.phone_noise_ratio,
        h.floor_noise_level,
    ).eval().requires_grad_(False)
    net_g.load_state_dict(ckpt["net_g"])
    net_g = net_g.to(device)

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
    """
    Convert input_path → output_path using the Beatrice 2 model in model_dir.
    model_dir should be a paraphernalia_*/ directory OR a directory containing
    a checkpoint_latest.pt.gz file (direct trainer output).
    """
    model_dir = Path(model_dir)

    # Prefer checkpoint.pt.gz (full precision, training-compatible) over .bin files
    ckpt = model_dir / "checkpoint_latest.pt.gz"
    if not ckpt.exists():
        # Try sibling directory (they live next to paraphernalia_* dirs)
        ckpt = model_dir.parent / "checkpoint_latest.pt.gz"
    
    if ckpt.exists():
        print(f"Using full checkpoint: {ckpt}")
        net_g, h = _load_checkpoint_model(ckpt, device)
        in_sr = h.in_sample_rate   # 16000
        out_sr = h.out_sample_rate  # 24000
    else:
        raise FileNotFoundError(
            f"No checkpoint_latest.pt.gz found in {model_dir} or {model_dir.parent}.\n"
            f"The paraphernalia .bin files use a custom binary format that requires the "
            f"closed beatrice.lib runtime. Use the .pt.gz checkpoint for PyTorch inference.\n"
            f"The checkpoint is saved alongside the paraphernalia_*/ dirs in the output_dir "
            f"you specified when running beatrice_trainer."
        )

    # Load + resample input audio
    wav, sr = torchaudio.load(str(input_path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)  # stereo → mono
    if sr != in_sr:
        wav = torchaudio.functional.resample(wav, sr, in_sr)

    wav = wav.to(device)  # [1, T]

    # Run conversion
    with torch.inference_mode():
        t_id = torch.tensor([speaker_id], dtype=torch.long, device=device)
        f_shift = torch.tensor([formant_shift_semitones], dtype=torch.float, device=device)
        p_shift = torch.tensor([pitch_shift_semitones], dtype=torch.float, device=device)

        y_hat = net_g(
            wav.unsqueeze(0),          # [1, 1, T]
            t_id,
            f_shift,
            pitch_shift_semitone=p_shift,
        )  # [1, T_out]

    y_hat = y_hat.squeeze(0).cpu().float()  # [T_out]
    # Normalise to prevent clipping
    peak = y_hat.abs().max()
    if peak > 0.95:
        y_hat = y_hat * (0.95 / peak)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), y_hat.unsqueeze(0), out_sr)
    print(f"Saved: {output_path}  ({out_sr} Hz, {y_hat.shape[0] / out_sr:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Beatrice 2 offline voice conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Path to training output dir (contains checkpoint_latest.pt.gz)")
    parser.add_argument("-i", "--input", required=True,
                        help="Input audio file (wav, mp3, flac, …)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output wav path")
    parser.add_argument("-s", "--speaker", type=int, default=0,
                        help="Target speaker ID (0-indexed, matches training data dir order)")
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
        model_dir=Path(args.model),
        input_path=Path(args.input),
        output_path=Path(args.output),
        speaker_id=args.speaker,
        pitch_shift_semitones=args.pitch_shift,
        formant_shift_semitones=args.formant_shift,
        device=device,
    )


if __name__ == "__main__":
    main()
