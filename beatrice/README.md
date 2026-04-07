# Beatrice 2 — Fine-tuning & Offline Conversion

Minimal setup for training and testing Beatrice 2 voice cloning within the rvc-web project.

---

## What's here

```
beatrice/
  train.sh            Shell wrapper — fine-tune a model (calls beatrice_trainer)
  convert.py          Offline converter — input.wav → cloned_output.wav
  download_assets.py  Downloads pretrained PhoneExtractor/PitchEstimator weights from HF
  trainer/            git clone of fierce-cats/beatrice-trainer (MIT)
    beatrice_trainer/
      __main__.py     Patched trainer (MPS + CPU support, fp32 on non-CUDA)
    assets/
      pretrained/     Downloaded by download_assets.py (~175 MB total)
      ir/             Room impulse responses (from LFS)
      noise/          Background noise files (from LFS)
      test/           Japanese test utterances
      default_config.json
```

---

## Prerequisites

- The `rvc` conda environment (already has torch, torchaudio, pyworld, soundfile, tensorboard)
- No additional dependencies needed

Verify:
```bash
conda run -n rvc python3 -c "import torch, torchaudio, pyworld, soundfile; print('OK')"
```

---

## Step 1 — Download pretrained base models

The trainer requires pretrained PhoneExtractor and PitchEstimator weights (~175 MB):

```bash
conda activate rvc
python3 beatrice/download_assets.py --trainer-dir beatrice/trainer
```

This only needs to run once.

---

## Step 2 — Prepare training data

Create a directory with one sub-directory per speaker:

```
my_data/
  alice/
    001.wav
    002.wav
    ...
```

- **Minimum**: ~5–10 minutes of clean speech (more is better, up to ~30 min)
- **Format**: wav/mp3/flac, any sample rate (trainer resamples internally to 16 kHz)
- **Quality**: Clean recordings preferred; the trainer applies noise augmentation
- **Language**: Any language (PhoneExtractor is language-agnostic)

---

## Step 3 — Fine-tune

```bash
conda activate rvc

# Mac (MPS, ~1.5–3 hours for 10k steps):
./beatrice/train.sh -d my_data/ -o output/beatrice_alice/

# Ubuntu / A100 CUDA (~30 min for 10k steps):
./beatrice/train.sh -d my_data/ -o output/beatrice_alice/

# Faster test run (5000 steps, smaller batch):
./beatrice/train.sh -d my_data/ -o output/beatrice_alice/ -n 5000 -b 4

# Resume after interruption:
./beatrice/train.sh -d my_data/ -o output/beatrice_alice/ -r
```

Training saves checkpoints every 2000 steps. The file you need for conversion is:
```
output/beatrice_alice/checkpoint_latest.pt.gz
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir output/beatrice_alice/
```

---

## Step 4 — Offline conversion

```bash
conda activate rvc

python3 beatrice/convert.py \
  --model  output/beatrice_alice/ \
  --input  source_audio.wav \
  --output converted_output.wav \
  --speaker 0

# With pitch shift (semitones):
python3 beatrice/convert.py \
  --model  output/beatrice_alice/ \
  --input  source_audio.wav \
  --output converted_output.wav \
  --speaker 0 \
  --pitch-shift -3

# Force a specific device:
python3 beatrice/convert.py --device mps ...   # Mac GPU
python3 beatrice/convert.py --device cuda ...  # NVIDIA GPU
python3 beatrice/convert.py --device cpu ...   # CPU fallback
```

**Output**: 24 kHz mono WAV. The model was trained at 24 kHz — this is Beatrice's output rate.

**Speaker IDs**: If you trained with multiple speakers (multiple subdirs in my_data/), use
`--speaker 0`, `--speaker 1`, etc. to select among them. The order matches the alphabetical
order of subdirectory names.

---

## Cross-platform notes

| Platform | Device | AMP | Expected quality |
|---|---|---|---|
| Ubuntu + A100 | cuda | fp32 (stable, default) | Full |
| Mac Apple Silicon | mps | fp32 (MPS fp16 has same overflow issues as RVC RefineGAN) | Full |
| CPU (any) | cpu | fp32 | Full (slow) |

The MPS patches in `beatrice_trainer/__main__.py`:
1. Device detection: prefers `mps` over `cpu` when CUDA unavailable
2. AMP disabled on non-CUDA (same rationale as RVC fix: MPS fp16 overflows)
3. `torch.cuda.empty_cache()` guarded by `torch.cuda.is_available()`

---

## Architecture summary

Beatrice 2 = StreamVC + per-speaker fine-tuning:

```
Source audio (16 kHz)
    ↓
[PhoneExtractor]  causal conv + self-attention → soft phone units
    + kNN-VC quantisation → blocks source timbre leakage
[PitchEstimator]  causal conv → whitened F0
    → avoids timbre/pitch coupling (RVC weakness)
[WaveformGenerator]  cross-attention speaker conditioning
    + FIRNet post-filter + D4C aperiodicity loss
    ↓
Output audio (24 kHz)
```

- 38ms algorithmic latency (designed for streaming)
- RTF < 0.2 on a mid-range CPU (no GPU needed at inference)
- ~30 MB model size after training

---

## Next steps after quality evaluation

If Beatrice quality exceeds RVC for your use case:

1. **VCClient sidecar** (easiest, Windows/Mac): use trained `paraphernalia_*/` dir with VCClient
2. **Pure PyTorch integration** (Linux CUDA, no closed lib): the `convert.py` path already does
   this — the same approach can be wired into rvc-web's realtime backend
