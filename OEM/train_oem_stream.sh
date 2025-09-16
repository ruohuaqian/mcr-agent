#!/usr/bin/env bash

set -euo pipefail

# ===================== Defaults =====================
# --- Paths ---
MCR_ROOT=${MCR_ROOT:-/content/mcr-agent}
DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/MyDrive}

SPLITS_DEFAULT_JSON=${DRIVE_ROOT}/mcr-agent/data/splits/oct21.json
DATA_DEFAULT_LOCAL=${DRIVE_ROOT}/mcr-agent/data/json_feat_2.1.0
DOUT_DEFAULT=${DRIVE_ROOT}/mcr-agent/exp/OEM_Training

# --- Hyperparameters (aligned with Python's argparse) ---
SEED=123
# For streaming mode
HUGGINGFACE_ID="byeonghwikim/abp_dataset"
SPLITS="$SPLITS_DEFAULT_JSON"
PP_FOLDER="pp"
SAVE_EVERY_EPOCH=0
MODEL="OEM.models.model.seq2seq_im_mask"
GPU=1
DOUT="$DOUT_DEFAULT"
RESUME=""
USE_TEMPLATED_GOALS=0
USE_STREAMING=1  # Default to streaming mode

BATCH=32 # Reduced default for better stability
EPOCH=40
LR=1e-4
DECAY_EPOCH=10
DHID=512
DFRAME=$((3*7*7))
DEMB=100
PFRAME=300
MASK_LOSS_WT=1.0
ACTION_LOSS_WT=1.0
SUBGOAL_AUX_LOSS_WT=0.2
PM_AUX_LOSS_WT=0.2

ZERO_GOAL=0
ZERO_INSTR=0
LANG_DROPOUT=0.0
INPUT_DROPOUT=0.0
VIS_DROPOUT=0.3
HSTATE_DROPOUT=0.3
ATTN_DROPOUT=0.0
ACTOR_DROPOUT=0.0

DEC_TEACHER_FORCING=0
TEMP_NO_HISTORY=0
PANORAMIC=1
ORIENTATION=1
PANORAMIC_CONCAT=1

FAST_EPOCH=0
DATASET_FRACTION=0
duration=0 # Initialize duration to prevent unbound variable error

# ===================== CLI Parameter Parsing =====================
while [[ $# -gt 0 ]]; do
  case "$1" in
    # Path and Mode controls
    --seed) SEED="$2"; shift 2;;
    --huggingface_id) HUGGINGFACE_ID="$2"; shift 2;;
    --splits) SPLITS="$2"; shift 2;;
    --pp_folder) PP_FOLDER="$2"; shift 2;;
    --save_every_epoch) SAVE_EVERY_EPOCH=1; shift;;
    --model) MODEL="$2"; shift 2;;
    --gpu) GPU=1; shift;;
    --no-gpu) GPU=0; shift;;
    --dout) DOUT="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --use_templated_goals) USE_TEMPLATED_GOALS=1; shift;;
    --use_streaming) USE_STREAMING=1; shift;;
    --no-streaming) USE_STREAMING=0; shift;;

    # Hyperparameters
    --batch) BATCH="$2"; shift 2;;
    --epoch) EPOCH="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --dhid) DHID="$2"; shift 2;;
    # ... Add other hyperparameter parsing here if needed ...

    # Debugging
    --fast_epoch) FAST_EPOCH=1; shift;;
    --dataset_fraction) DATASET_FRACTION="$2"; shift 2;;

    # Help
    --help|-h)
      echo "Usage: $0 [options]"
      echo ""
      echo "Mode Control:"
      echo "  --use_streaming       Enable Hugging Face streaming mode (default)."
      echo "  --no-streaming        Disable streaming, use local file mode."
      echo ""
      echo "Key Paths:"
      echo "  --huggingface_id ID   Hugging Face dataset ID for streaming (default: $HUGGINGFACE_ID)."
      echo "  --splits PATH         Path to splits JSON file (default: $SPLITS_DEFAULT_JSON)."
      echo "  --dout PATH           Path to save model output (default: $DOUT_DEFAULT)."
      echo ""
      echo "Common Training Parameters:"
      echo "  --batch BATCH         Batch size (default: $BATCH)."
      echo "  --epoch EPOCH         Number of epochs (default: $EPOCH)."
      echo "  --lr LR               Learning rate (default: $LR)."
      echo "  --fast_epoch          Run a very small subset for debugging."
      exit 0;;
    *) echo "[WARN] Unknown option: $1, ignoring." ; shift;;
  esac
done

# ===================== Environment Setup =====================
# Ensure output directory exists
mkdir -p "$DOUT"

# Set project root and PYTHONPATH
export MCR_ROOT="$MCR_ROOT"
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/OEM:${PYTHONPATH}"

# Navigate to the correct subdirectory to run the script
cd "${MCR_ROOT}/OEM"

# Auto-install dependencies for streaming mode if needed
if [[ "$USE_STREAMING" -eq 1 ]]; then
  if ! python -c "import datasets, huggingface_hub, requests" &>/dev/null; then
    echo "[INFO] Installing streaming dependencies (datasets, huggingface-hub, requests)..."
    pip install datasets huggingface-hub requests --quiet
  fi
fi

# ===================== Assemble Python Command =====================

CMD=( python "models/train/train_seq2seq_stream.py"
  --seed "$SEED"
  --splits "$SPLITS"
  --pp_folder "$PP_FOLDER"
  --huggingface_id "$HUGGINGFACE_ID"
  --model "$MODEL"
  --dout "$DOUT"
  --batch "$BATCH"
  --epoch "$EPOCH"
  --lr "$LR"
  --decay_epoch "$DECAY_EPOCH"
  --dhid "$DHID"
  --dframe "$DFRAME"
  --demb "$DEMB"
  --pframe "$PFRAME"
  --mask_loss_wt "$MASK_LOSS_WT"
  --action_loss_wt "$ACTION_LOSS_WT"
  --subgoal_aux_loss_wt "$SUBGOAL_AUX_LOSS_WT"
  --pm_aux_loss_wt "$PM_AUX_LOSS_WT"
  --lang_dropout "$LANG_DROPOUT"
  --input_dropout "$INPUT_DROPOUT"
  --vis_dropout "$VIS_DROPOUT"
  --hstate_dropout "$HSTATE_DROPOUT"
  --attn_dropout "$ATTN_DROPOUT"
  --actor_dropout "$ACTOR_DROPOUT"
)


# Add optional boolean flags
[[ "$SAVE_EVERY_EPOCH" -eq 1 ]] && CMD+=( --save_every_epoch )
[[ "$GPU" -eq 1 ]] && CMD+=( --gpu )
[[ -n "$RESUME" ]] && CMD+=( --resume "$RESUME" )
[[ "$USE_TEMPLATED_GOALS" -eq 1 ]] && CMD+=( --use_templated_goals )
[[ "$USE_STREAMING" -eq 1 ]] && CMD+=( --use_streaming )
[[ "$ZERO_GOAL" -eq 1 ]] && CMD+=( --zero_goal )
[[ "$ZERO_INSTR" -eq 1 ]] && CMD+=( --zero_instr )
[[ "$DEC_TEACHER_FORCING" -eq 1 ]] && CMD+=( --dec_teacher_forcing )
[[ "$TEMP_NO_HISTORY" -eq 1 ]] && CMD+=( --temp_no_history )
[[ "$PANORAMIC" -eq 1 ]] && CMD+=( --panoramic )
[[ "$ORIENTATION" -eq 1 ]] && CMD+=( --orientation )
[[ "$PANORAMIC_CONCAT" -eq 1 ]] && CMD+=( --panoramic_concat )
[[ "$FAST_EPOCH" -eq 1 ]] && CMD+=( --fast_epoch )
[[ "$DATASET_FRACTION" -gt 0 ]] && CMD+=( --dataset_fraction "$DATASET_FRACTION" )

# ===================== Execute Training =====================
echo "========================================================================"
echo "[INFO] Starting Training Run"
echo "------------------------------------------------------------------------"
echo "[INFO] Mode: $(if [[ "$USE_STREAMING" -eq 1 ]]; then echo "Streaming"; else echo "Local File"; fi)"
echo "[INFO] Model: $MODEL"
echo "[INFO] Output Dir: $DOUT"
echo "[INFO] Batch Size: $BATCH"
echo "[INFO] Epochs: $EPOCH"
echo "------------------------------------------------------------------------"
echo "[CMD] ${CMD[*]}"
echo "========================================================================"

start_time=$(date +%s)

if ! "${CMD[@]}"; then
    echo "[ERROR] Training script failed with exit code $?."
    exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "[SUCCESS] Training completed in $((duration / 60))m $((duration % 60))s."

# ===================== Create Training Summary =====================
SUMMARY_FILE="${DOUT}/training_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Training Summary
================
Date: $(date)
Model: $MODEL
Mode: $(if [[ "$USE_STREAMING" -eq 1 ]]; then echo "Streaming (Hugging Face ID: $HUGGINGFACE_ID)"; else echo "Local File (Path: $DATA)"; fi)
Output Directory: $DOUT

Hyperparameters:
- Seed: $SEED
- Batch size: $BATCH
- Epochs: $EPOCH
- Learning rate: $LR
- Hidden size: $DHID

Training Time: $((duration / 60)) minutes $((duration % 60)) seconds
EOF

echo "[INFO] Training summary saved to: $SUMMARY_FILE"
echo "========================================================================"