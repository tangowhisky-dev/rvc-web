#!/usr/bin/env python3
"""
beatrice/download_assets.py — download pretrained Beatrice 2 base models from HuggingFace.

These are the PhoneExtractor, PitchEstimator, and pretrained WaveformGenerator
checkpoints that beatrice_trainer loads at the start of fine-tuning.

Usage:
    python3 beatrice/download_assets.py [--trainer-dir beatrice/trainer]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/fierce-cats/beatrice-trainer/resolve/main"

ASSETS = [
    # (relative_path, size_bytes_approx)
    ("assets/pretrained/104_3_checkpoint_00300000.pt", 7_100_000),   # PitchEstimator
    ("assets/pretrained/122_checkpoint_03000000.pt",   14_700_000),  # PhoneExtractor
    ("assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz", 153_200_000),  # WaveformGenerator base
    # Test files (used for evaluation audio during training)
    ("assets/test/common_voice_ja_38833628_16k.wav",  200_000),
    ("assets/test/common_voice_ja_38843402_16k.wav",  200_000),
    ("assets/test/common_voice_ja_38852485_16k.wav",  300_000),
    ("assets/test/common_voice_ja_38853932_16k.wav",  200_000),
    ("assets/test/common_voice_ja_38864552_16k.wav",  300_000),
    ("assets/test/common_voice_ja_38878413_16k.wav",  300_000),
    ("assets/test/common_voice_ja_38898180_16k.wav",  300_000),
    ("assets/test/common_voice_ja_38925334_16k.wav",  300_000),
]


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"  Downloading {dest.name} ...", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  Downloading {dest.name} ... {pct}%", end="", flush=True)
        tmp.rename(dest)
        print(f"\r  Downloaded  {dest.name} ({downloaded / 1e6:.1f} MB)")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_assets(trainer_dir: Path, force: bool = False):
    print(f"Trainer directory: {trainer_dir}")
    for rel_path, _ in ASSETS:
        dest = trainer_dir / rel_path
        if dest.exists() and not force:
            print(f"  Exists: {rel_path}")
            continue
        url = f"{HF_BASE}/{rel_path}"
        _download(url, dest)
    print("All assets ready.")


def main():
    parser = argparse.ArgumentParser(description="Download Beatrice 2 pretrained assets")
    parser.add_argument("--trainer-dir", default="beatrice/trainer",
                        type=Path, help="Path to beatrice/trainer directory")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    trainer_dir = args.trainer_dir.resolve()
    if not (trainer_dir / "beatrice_trainer/__main__.py").exists():
        print(f"Error: beatrice_trainer/__main__.py not found in {trainer_dir}", file=sys.stderr)
        print("Run: cd beatrice/trainer && git lfs pull", file=sys.stderr)
        sys.exit(1)

    download_assets(trainer_dir, force=args.force)


if __name__ == "__main__":
    main()
