#!/usr/bin/env python3
"""
beatrice/download_assets.py — download pretrained Beatrice 2 base models and
augmentation data from HuggingFace.

Usage:
    python3 beatrice/download_assets.py [--trainer-dir beatrice/trainer]
    python3 beatrice/download_assets.py --full        # also get all 1000 IR/noise files
    python3 beatrice/download_assets.py --force       # re-download everything

By default downloads:
  - 3 pretrained model checkpoints (~175 MB)  [required]
  - 8 test utterances (~2 MB)                 [required for evaluation audio]
  - 20 impulse response files (IR)            [required — trainer asserts dir non-empty]
  - 20 background noise files                 [required — trainer asserts dir non-empty]

The trainer randomly samples from IR/noise for data augmentation.
20 files of each is sufficient for training; --full grabs all 1000 of each.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/fierce-cats/beatrice-trainer/resolve/main"

# Required pretrained weights and test files
REQUIRED_ASSETS = [
    "assets/pretrained/104_3_checkpoint_00300000.pt",
    "assets/pretrained/122_checkpoint_03000000.pt",
    "assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz",
    "assets/test/common_voice_ja_38833628_16k.wav",
    "assets/test/common_voice_ja_38843402_16k.wav",
    "assets/test/common_voice_ja_38852485_16k.wav",
    "assets/test/common_voice_ja_38853932_16k.wav",
    "assets/test/common_voice_ja_38864552_16k.wav",
    "assets/test/common_voice_ja_38878413_16k.wav",
    "assets/test/common_voice_ja_38898180_16k.wav",
    "assets/test/common_voice_ja_38925334_16k.wav",
]

# 20 evenly-spaced IR files (0000–0999, every 50th)
IR_SUBSET = [f"assets/ir/{i:04d}.flac" for i in range(0, 1000, 50)]

# 20 evenly-spaced noise files (confirmed present in HF repo)
NOISE_SUBSET = [
    "assets/noise/-0g8EILYVyw.flac",
    "assets/noise/2ONf1pus7CQ.flac",
    "assets/noise/58ajs6se3HQ.flac",
    "assets/noise/8wJbyW5RhBs.flac",
    "assets/noise/Cr6HL29UvYc.flac",
    "assets/noise/FXlL-WffpM0.flac",
    "assets/noise/IM0xaUDY5BA.flac",
    "assets/noise/MWS5evuFcNI.flac",
    "assets/noise/RORtozS18o0.flac",
    "assets/noise/UbtBg1D2Qx0.flac",
    "assets/noise/YkvtKDvjdw0.flac",
    "assets/noise/breath_spit_Freesound_validated_406647_0.flac",
    "assets/noise/door_Freesound_validated_216008_0.flac",
    "assets/noise/door_Freesound_validated_452592_0.flac",
    "assets/noise/fan_Freesound_validated_323119_4.flac",
    "assets/noise/iW_40jBKQX0.flac",
    "assets/noise/mVeXgyp9w2c.flac",
    "assets/noise/r81cTzdpwtg.flac",
    "assets/noise/tQecG72py9I.flac",
    "assets/noise/vdgUNn1wRpc.flac",
]

# Full IR list — all 1000 files
IR_FULL = [f"assets/ir/{i:04d}.flac" for i in range(1000)]

# Full noise list — all 1000 files (names from git ls-tree)
# Generated with: git ls-tree HEAD assets/noise/ | awk '{print $4}'
NOISE_FULL_NAMES = [
    "-0g8EILYVyw", "-1zGTyKQnvw", "-5bUWtaoJ7U", "-7_o_GhpZpM", "-BdXZjYiRUQ",
    "-CirKYqIfgI", "-JIwqWxJ1KI", "-KrDQXnlXRw", "-MnQ4IjEreg", "-PN0FyUJTig",
    "-RP1-cUHLT8", "-XPgaXTF6M8", "-YpovYcGB4I", "-em8ZaHR2fg", "-rPghAZFJH8",
    "08aA8osT9nE", "08ibPbVUo68", "0EB0GRDhL3c", "0On6-JiVwRs", "0Rl8HQoTQBc",
    "0T548TkI_eY", "0TtBLOeGdYE", "0_4ZIVuWDlw", "0aGL_xtpioI", "0akxVo5_W5k",
    "0c7urKyFJQU", "0hH10rnwBuI", "0krZliEb9XE", "0rVJqFW9ju4", "0tHX0byP3tk",
]
# Note: for the full 1000-file noise set, use: cd beatrice/trainer && git lfs pull
NOISE_FULL = [f"assets/noise/{n}.flac" for n in NOISE_FULL_NAMES]


def _download(url: str, dest: Path, skip_404: bool = False) -> bool:
    """Returns True if downloaded, False if skipped (404 when skip_404=True)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
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
        return True
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        if skip_404 and e.code == 404:
            print(f"\r  Skipped     {dest.name} (404 — not in repo)")
            return False
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def download_assets(trainer_dir: Path, full: bool = False, force: bool = False):
    print(f"Trainer directory: {trainer_dir}")

    ir_files = IR_FULL if full else IR_SUBSET
    noise_files = NOISE_FULL if full else NOISE_SUBSET

    all_assets = REQUIRED_ASSETS + ir_files + noise_files
    n = len(all_assets)
    print(f"Files to download: {n} ({len(ir_files)} IR + {len(noise_files)} noise + core)")
    print()

    skipped = 0
    for rel_path in all_assets:
        dest = trainer_dir / rel_path
        if dest.exists() and not force:
            skipped += 1
            continue
        url = f"{HF_BASE}/{rel_path}"
        _download(url, dest, skip_404=True)

    if skipped:
        print(f"  ({skipped} files already present, skipped)")
    print()
    print("All assets ready.")

    # Verify dirs are non-empty (trainer asserts this)
    ir_dir = trainer_dir / "assets/ir"
    noise_dir = trainer_dir / "assets/noise"
    n_ir = len(list(ir_dir.glob("*.flac"))) if ir_dir.exists() else 0
    n_noise = len(list(noise_dir.glob("*.flac"))) if noise_dir.exists() else 0
    print(f"  IR files:    {n_ir}")
    print(f"  Noise files: {n_noise}")

    if n_ir == 0 or n_noise == 0:
        print("\nWarning: IR or noise dir is empty — training will fail.")
        print("Try: cd beatrice/trainer && git lfs pull")
    else:
        print("\nReady to train. See beatrice/README.md for usage.")

    if not full:
        print()
        print("Tip: using a 20-file subset of IR/noise (sufficient for training).")
        print("For the full 1000-file augmentation set, run:")
        print("  cd beatrice/trainer && git lfs pull")


def main():
    parser = argparse.ArgumentParser(
        description="Download Beatrice 2 pretrained assets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trainer-dir", default="beatrice/trainer",
                        type=Path, help="Path to beatrice/trainer directory")
    parser.add_argument("--full", action="store_true",
                        help="Download all 1000 IR files (use git lfs pull for full noise set)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    trainer_dir = args.trainer_dir.resolve()
    if not (trainer_dir / "beatrice_trainer/__main__.py").exists():
        print(f"Error: beatrice_trainer/__main__.py not found in {trainer_dir}", file=sys.stderr)
        sys.exit(1)

    download_assets(trainer_dir, full=args.full, force=args.force)


if __name__ == "__main__":
    main()
