#!/usr/bin/env bash
# beatrice/train.sh — fine-tune a Beatrice 2 model using the conda rvc environment
#
# Usage (from any directory):
#   ./beatrice/train.sh -d my_data/ -o output/beatrice_alice/ [options]
#
# Arguments:
#   -d DIR       Directory with target speaker audio. Must contain one subdir
#                per speaker, e.g.:
#                  my_data/alice/recording.mp3
#                Any audio format (wav, mp3, flac, m4a, …) is accepted.
#                Long files (>15s) are automatically split into 8s WAV chunks
#                before training. Originals are moved to _originals/ subfolders.
#   -o DIR       Output directory (checkpoints saved here)
#   -r           Resume from checkpoint_latest.pt.gz in output dir
#   -c FILE      Path to custom config JSON
#   -n INT       Number of training steps (default: 10000)
#   -b INT       Batch size (default: 16 for CUDA; try 8 if OOM)
#   --skip-preprocess  Skip the preprocessing step (data already chunked)
#
# Examples:
#   ./beatrice/train.sh -d my_data/ -o output/beatrice_alice/ -n 10000 -b 16
#   ./beatrice/train.sh -d my_data/ -o output/beatrice_alice/ -r          # resume

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
SKIP_PREPROCESS=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DATA_DIR="$2"; shift 2;;
        -o) OUT_DIR="$2"; shift 2;;
        -r) RESUME="--resume"; shift;;
        -c) CONFIG="$2"; shift 2;;
        -n) N_STEPS="$2"; shift 2;;
        -b) BATCH_SIZE="$2"; shift 2;;
        --skip-preprocess) SKIP_PREPROCESS="1"; shift;;
        --device) shift 2;;  # ignored — trainer auto-detects
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

if [[ -z "$DATA_DIR" || -z "$OUT_DIR" ]]; then
    echo "Error: -d DATA_DIR and -o OUT_DIR are required."
    exit 1
fi

# ---- resolve to absolute paths before cd-ing into trainer dir ----
DATA_DIR="$(cd "$(dirname "$DATA_DIR")" && pwd)/$(basename "$DATA_DIR")"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# ---- sanity check trainer ----
if [[ ! -f "$TRAINER_DIR/beatrice_trainer/__main__.py" ]]; then
    echo "Trainer not found at $TRAINER_DIR/beatrice_trainer/__main__.py"
    exit 1
fi

if [[ ! -f "$TRAINER_DIR/assets/pretrained/122_checkpoint_03000000.pt" ]]; then
    echo "Pretrained assets not found. Downloading..."
    python3 "$SCRIPT_DIR/download_assets.py"
fi

# ---- preprocess: split long files into 8s chunks ----
if [[ -z "$SKIP_PREPROCESS" ]]; then
    echo "Preprocessing: splitting long audio files into 8s chunks..."
    python3 "$SCRIPT_DIR/preprocess.py" --data-dir "$DATA_DIR"
    echo ""
fi

# ---- build config override if -n or -b set ----
TMP_CFG=""
EXTRA_CONFIG_ARG=""
if [[ -n "$N_STEPS" || -n "$BATCH_SIZE" ]]; then
    TMP_CFG=$(mktemp /tmp/beatrice_cfg.XXXXXX)
    trap 'rm -f "$TMP_CFG"' EXIT
    cp "$TRAINER_DIR/assets/default_config.json" "$TMP_CFG"
    python3 - "$TMP_CFG" "$N_STEPS" "$BATCH_SIZE" <<'PYEOF'
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
        echo "Warning: -c and -n/-b both set; using -c, ignoring -n/-b"
        rm -f "$TMP_CFG"; TMP_CFG=""
    else
        EXTRA_CONFIG_ARG="--config $TMP_CFG"
    fi
fi

CONFIG_ARG=""
[[ -n "$CONFIG" ]] && CONFIG_ARG="--config $(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

# ---- run ----
echo "============================================================"
echo "Beatrice 2 fine-tuning"
echo "  data:   $DATA_DIR"
echo "  output: $OUT_DIR"
echo "  steps:  ${N_STEPS:-10000 (default)}"
echo "  batch:  ${BATCH_SIZE:-16 (default)}"
echo "============================================================"

cd "$TRAINER_DIR"
export BEATRICE_TRAINER_ROOT="$TRAINER_DIR"
python3 -m beatrice_trainer \
    --data_dir "$DATA_DIR" \
    --out_dir  "$OUT_DIR" \
    $RESUME \
    $CONFIG_ARG \
    $EXTRA_CONFIG_ARG

echo ""
echo "Training complete. Outputs in: $OUT_DIR"
echo ""
echo "Convert audio:"
echo "  conda run -n rvc python3 $SCRIPT_DIR/convert.py \\"
echo "    --model $OUT_DIR --input input.wav --output out.wav --speaker 0"
