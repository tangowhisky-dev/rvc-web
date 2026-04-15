#!/usr/bin/env python3
"""
backend/beatrice2/download_assets.py — download pretrained Beatrice 2 base models and
augmentation data from HuggingFace into assets/beatrice2/.

Usage (from rvc-web/ root):
    python3 -m backend.beatrice2.download_assets
    python3 -m backend.beatrice2.download_assets --full        # all 1000 IR/noise files
    python3 -m backend.beatrice2.download_assets --force       # re-download everything

By default downloads:
  - 3 pretrained model checkpoints (~175 MB)  [required]
  - 8 test utterances (~2 MB)                 [required for evaluation audio]
  - 20 impulse response files (IR)            [required — trainer asserts dir non-empty]
  - 20 background noise files                 [required — trainer asserts dir non-empty]

All files are placed under {PROJECT_ROOT}/assets/beatrice2/ which mirrors the
HF repo layout (assets/pretrained/, assets/ir/, assets/noise/, assets/test/).
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


def _project_root() -> Path:
    """Return rvc-web/ project root."""
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env)
    # Fallback: two levels up from this file (backend/beatrice2/ → rvc-web/)
    return Path(__file__).resolve().parent.parent.parent


def _default_assets_dir() -> Path:
    return _project_root() / "assets" / "beatrice2"


def download_assets(assets_dir: Path, full: bool = False, force: bool = False):
    """Download all required Beatrice 2 assets into *assets_dir*.

    The HF repo layout mirrors our target layout exactly:
      assets/pretrained/*, assets/ir/*, assets/noise/*, assets/test/*
    so each rel_path is appended directly to assets_dir.
    """
    print(f"Assets directory: {assets_dir}")

    ir_files = IR_FULL if full else IR_SUBSET
    noise_files = NOISE_FULL if full else NOISE_SUBSET

    all_assets = REQUIRED_ASSETS + ir_files + noise_files
    n = len(all_assets)
    print(f"Files to download: {n} ({len(ir_files)} IR + {len(noise_files)} noise + core)")
    print()

    skipped = 0
    for rel_path in all_assets:
        # rel_path starts with "assets/" — strip that prefix so files land
        # directly under assets_dir (e.g. assets_dir/pretrained/...)
        stripped = rel_path[len("assets/"):]
        dest = assets_dir / stripped
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
    ir_dir = assets_dir / "ir"
    noise_dir = assets_dir / "noise"
    n_ir = len(list(ir_dir.glob("*.flac"))) if ir_dir.exists() else 0
    n_noise = len(list(noise_dir.glob("*.flac"))) if noise_dir.exists() else 0
    print(f"  IR files:    {n_ir}")
    print(f"  Noise files: {n_noise}")

    if n_ir == 0 or n_noise == 0:
        print("\nWarning: IR or noise dir is empty — training will fail.")
    else:
        print("\nReady to train.")

    if not full:
        print()
        print("Tip: using a 20-file subset of IR/noise (sufficient for training).")
        print("For the full 1000-file augmentation set, re-run with --full.")


def main():
    parser = argparse.ArgumentParser(
        description="Download Beatrice 2 pretrained assets into assets/beatrice2/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--assets-dir", default=None, type=Path,
        help="Destination directory (default: {PROJECT_ROOT}/assets/beatrice2/)",
    )
    parser.add_argument("--full", action="store_true",
                        help="Download all 1000 IR/noise files (default: 20-file subset)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    assets_dir = Path(args.assets_dir).resolve() if args.assets_dir else _default_assets_dir()
    download_assets(assets_dir, full=args.full, force=args.force)


if __name__ == "__main__":
    main()
