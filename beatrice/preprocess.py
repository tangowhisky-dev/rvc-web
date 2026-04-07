#!/usr/bin/env python3
"""
beatrice/preprocess.py — split long audio files into fixed-length chunks.

The Beatrice 2 trainer expects a directory of short utterances (~5–15s each),
one subdirectory per speaker.  This script scans the input directory, moves
any file longer than --max-duration seconds to a _originals/ subfolder, and
writes fixed-length WAV chunks in its place.

Usage:
    python3 beatrice/preprocess.py -d my_data/

Options:
    -d DIR            Data directory (required). Must contain one subdir per speaker.
    --chunk-duration  Chunk length in seconds (default: 4.0, matches trainer wav_length)
    --max-duration    Files longer than this are split (default: 4.0)
    --sample-rate     Output sample rate in Hz (default: 24000, matches trainer)
    --dry-run         Print what would happen without changing any files
"""

import argparse
import shutil
import sys
import warnings
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".fla", ".flac", ".oga", ".ogg",
                  ".opus", ".mp3", ".m4a", ".aac"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--data-dir", required=True, type=Path,
                   help="Data directory with one subdir per speaker")
    p.add_argument("--chunk-duration", type=float, default=4.0,
                   help="Chunk length in seconds (default: 4.0, matches trainer wav_length)")
    p.add_argument("--max-duration", type=float, default=4.0,
                   help="Files longer than this are split (default: 4.0)")
    p.add_argument("--sample-rate", type=int, default=24000,
                   help="Output sample rate (default: 24000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing files")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        import torchaudio
    except ImportError:
        print("Error: torch and torchaudio are required.", file=sys.stderr)
        sys.exit(1)

    # Suppress torchaudio warnings that aren't useful here
    warnings.filterwarnings("ignore", category=UserWarning)

    chunk_samples = int(args.chunk_duration * args.sample_rate)
    total_files = 0
    total_split = 0
    total_chunks = 0

    for speaker_dir in sorted(data_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue

        audio_files = sorted(
            f for f in speaker_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES
        )

        for src_file in audio_files:
            total_files += 1

            # Load to check duration
            try:
                wav, sr = torchaudio.load(str(src_file))
            except Exception as e:
                print(f"  SKIP (load error): {src_file.relative_to(data_dir)}: {e}")
                continue

            duration = wav.shape[-1] / sr

            if duration <= args.max_duration:
                # Short enough — leave it alone
                print(f"  ok  ({duration:.1f}s): {src_file.relative_to(data_dir)}")
                continue

            # Resample to target rate if needed
            if sr != args.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, args.sample_rate)
                wav = resampler(wav)

            # Mix to mono
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            n_samples = wav.shape[-1]
            n_chunks = n_samples // chunk_samples
            if n_chunks == 0:
                print(f"  SKIP (too short after resample): {src_file.relative_to(data_dir)}")
                continue

            originals_dir = speaker_dir / "_originals"
            stem = src_file.stem

            print(f"  split ({duration:.1f}s -> {n_chunks} x {args.chunk_duration:.0f}s chunks): "
                  f"{src_file.relative_to(data_dir)}")

            if args.dry_run:
                total_split += 1
                total_chunks += n_chunks
                continue

            # Move original to _originals/
            originals_dir.mkdir(exist_ok=True)
            dest_original = originals_dir / src_file.name
            shutil.move(str(src_file), str(dest_original))

            # Write chunks as WAV
            for i in range(n_chunks):
                start = i * chunk_samples
                chunk = wav[:, start:start + chunk_samples]
                out_path = speaker_dir / f"{stem}_chunk{i:04d}.wav"
                torchaudio.save(str(out_path), chunk, args.sample_rate)

            total_split += 1
            total_chunks += n_chunks

    print()
    print(f"Done. {total_files} files scanned, {total_split} split into "
          f"{total_chunks} chunks total.")
    if args.dry_run:
        print("(dry run — no files were written)")
    else:
        print("Original files moved to _originals/ subdirectories.")


if __name__ == "__main__":
    main()
