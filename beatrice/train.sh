#!/usr/bin/env bash
# beatrice/train.sh — fine-tune a Beatrice 2 model using the conda rvc environment
#
# Usage:
#   cd ~/code/rvc-web
#   ./beatrice/train.sh -d <audio_dir> -o <output_dir> [options]
#
# Arguments:
#   -d DIR       Directory containing target speaker audio (wav/mp3/flac)
#                Must contain a sub-directory named after the speaker, e.g.:
#                  my_data/alice/001.wav
#                  my_data/alice/002.wav
#   -o DIR       Output directory (checkpoints + paraphernalia saved here)
#   -r           Resume from checkpoint_latest.pt.gz in output dir
#   -c FILE      Path to custom config JSON (copy assets/default_config.json and edit)
#   -n INT       Number of training steps (default: 10000)
#   -b INT       Batch size (default: 8; reduce to 4 if OOM on GPU)
#   --device STR cuda | mps | cpu (auto-detected if omitted)
#
# Prerequisites:
#   conda activate rvc   (already has torch, torchaudio, pyworld, soundfile, tensorboard)
#
# Examples:
#   # Train on CUDA (Ubuntu A100):
#   ./beatrice/train.sh -d ~/data/my_speaker -o ~/models/beatrice_my_speaker
#
#   # Train on Mac MPS:
#   ./beatrice/train.sh -d ~/data/my_speaker -o ~/models/beatrice_my_speaker
#
#   # Resume:
#   ./beatrice/train.sh -d ~/data/my_speaker -o ~/models/beatrice_my_speaker -r
#
#   # Low VRAM (batch 4, 5000 steps):
#   ./beatrice/train.sh -d ~/data/my_speaker -o ~/models/beatrice_my_speaker -b 4 -n 5000

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$SCRIPT_DIR/trainer"

# ---- defaults ----
DATA_DIR=""
OUT_DIR=""
RESUME=""
CONFIG=""
N_STEPS=""
BATCH_SIZE=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DATA_DIR="$2"; shift 2;;
        -o) OUT_DIR="$2"; shift 2;;
        -r) RESUME="--resume"; shift;;
        -c) CONFIG="--config $2"; shift 2;;
        -n) N_STEPS="$2"; shift 2;;
        -b) BATCH_SIZE="$2"; shift 2;;
        --device) shift 2;;  # ignored - trainer auto-detects
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [[ -z "$DATA_DIR" || -z "$OUT_DIR" ]]; then
    echo "Error: -d DATA_DIR and -o OUT_DIR are required."
    exit 1
fi

# ---- sanity check trainer ----
if [[ ! -f "$TRAINER_DIR/beatrice_trainer/__main__.py" ]]; then
    echo "Trainer not found at $TRAINER_DIR/beatrice_trainer/__main__.py"
    echo "Run: cd beatrice/trainer && git lfs pull"
    exit 1
fi

if [[ ! -f "$TRAINER_DIR/assets/pretrained/122_checkpoint_03000000.pt" ]]; then
    echo "Pretrained assets not found. Downloading via HuggingFace..."
    python3 "$SCRIPT_DIR/download_assets.py" --trainer-dir "$TRAINER_DIR"
fi

# ---- build config override if n_steps or batch_size set ----
EXTRA_CONFIG_ARG=""
if [[ -n "$N_STEPS" || -n "$BATCH_SIZE" ]]; then
    TMP_CFG=$(mktemp /tmp/beatrice_cfg_XXXXXX.json)
    # Start from default config
    cp "$TRAINER_DIR/assets/default_config.json" "$TMP_CFG"
    # Patch n_steps and/or batch_size
    python3 - <<PYEOF "$TMP_CFG" "$N_STEPS" "$BATCH_SIZE"
import sys, json
cfg_path, n_steps, batch_size = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path) as f:
    cfg = json.load(f)
if n_steps:
    cfg["n_steps"] = int(n_steps)
if batch_size:
    cfg["batch_size"] = int(batch_size)
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF
    if [[ -n "$CONFIG" ]]; then
        echo "Warning: -c and -n/-b both set; n_steps/batch_size override will be ignored in favour of -c"
        rm "$TMP_CFG"
    else
        EXTRA_CONFIG_ARG="--config $TMP_CFG"
    fi
fi

# ---- run ----
echo "============================================================"
echo "Beatrice 2 fine-tuning"
echo "  data:   $DATA_DIR"
echo "  output: $OUT_DIR"
echo "  steps:  ${N_STEPS:-10000 (default)}"
echo "  batch:  ${BATCH_SIZE:-8 (default)}"
echo "============================================================"

mkdir -p "$OUT_DIR"

cd "$TRAINER_DIR"
python3 -m beatrice_trainer \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
    ${RESUME} \
    ${CONFIG} \
    ${EXTRA_CONFIG_ARG}

echo ""
echo "Training complete. Outputs in: $OUT_DIR"
echo ""
echo "To convert audio with the trained model:"
echo "  conda run -n rvc python3 $SCRIPT_DIR/convert.py \\"
echo "    --model $OUT_DIR \\"
echo "    --input input.wav \\"
echo "    --output output.wav \\"
echo "    --speaker 0"
