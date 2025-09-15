#!/usr/bin/env bash
set -euo pipefail

# ========= default =========
MCR_ROOT=${MCR_ROOT:-$(pwd)}
DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/MyDrive}
SPLITS_DEFAULT_JSON=${DRIVE_ROOT}/data/splits/oct21.json

# ========= default hyperparameters =========
SEED=123
SPLITS="$SPLITS_DEFAULT_JSON"
PP_FOLDER="pp"
SAVE_EVERY_EPOCH=0
MODEL="Interactions.models.model.seq2seq_im_mask"
GPU=1
DOUT="${MCR_ROOT}/exp/model:seq2seq_im_mask"
RESUME=""
USE_TEMPLATED_GOALS=0
USE_STREAMING=1
HUGGINGFACE_ID="byeonghwikim/abp_dataset"

BATCH=4
EPOCH=20
LR=1e-4
DECAY_EPOCH=10
DHID=512
DFRAME=$((3*7*7))
DEMB=100
PFRAME=300
MASK_LOSS_WT=1.0
ACTION_LOSS_WT=1.0
SUBGOAL_AUX_LOSS_WT=0.0
PM_AUX_LOSS_WT=0.0

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

# ========= parse command line arguments =========
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
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
    --huggingface_id) HUGGINGFACE_ID="$2"; shift 2;;

    --batch) BATCH="$2"; shift 2;;
    --epoch) EPOCH="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --decay_epoch) DECAY_EPOCH="$2"; shift 2;;
    --dhid) DHID="$2"; shift 2;;
    --dframe) DFRAME="$2"; shift 2;;
    --demb) DEMB="$2"; shift 2;;
    --pframe) PFRAME="$2"; shift 2;;
    --mask_loss_wt) MASK_LOSS_WT="$2"; shift 2;;
    --action_loss_wt) ACTION_LOSS_WT="$2"; shift 2;;
    --subgoal_aux_loss_wt) SUBGOAL_AUX_LOSS_WT="$2"; shift 2;;
    --pm_aux_loss_wt) PM_AUX_LOSS_WT="$2"; shift 2;;

    --zero_goal) ZERO_GOAL=1; shift;;
    --zero_instr) ZERO_INSTR=1; shift;;
    --lang_dropout) LANG_DROPOUT="$2"; shift 2;;
    --input_dropout) INPUT_DROPOUT="$2"; shift 2;;
    --vis_dropout) VIS_DROPOUT="$2"; shift 2;;
    --hstate_dropout) HSTATE_DROPOUT="$2"; shift 2;;
    --attn_dropout) ATTN_DROPOUT="$2"; shift 2;;
    --actor_dropout) ACTOR_DROPOUT="$2"; shift 2;;

    --dec_teacher_forcing) DEC_TEACHER_FORCING=1; shift;;
    --temp_no_history) TEMP_NO_HISTORY=1; shift;;
    --panoramic) PANORAMIC=1; shift;;
    --orientation) ORIENTATION=1; shift;;
    --panoramic_concat) PANORAMIC_CONCAT=1; shift;;

    --fast_epoch) FAST_EPOCH=1; shift;;
    --dataset_fraction) DATASET_FRACTION="$2"; shift 2;;

    --help|-h)
      echo "Usage: $0 [--huggingface_id ID] [--splits PATH] [--dout PATH] [--model NAME] [--use_streaming] ..."
      echo ""
      echo "Subgoal Training Options:"
      echo "  --huggingface_id      Hugging Face dataset ID (default: $HUGGINGFACE_ID)"
      echo "  --use_streaming       Enable Hugging Face streaming mode (default: enabled)"
      echo "  --no-streaming        Disable streaming mode"
      echo ""
      echo "Training Parameters:"
      echo "  --batch               Batch size (default: $BATCH)"
      echo "  --epoch               Number of epochs (default: $EPOCH)"
      echo "  --lr                  Learning rate (default: $LR)"
      echo "  --dhid                Hidden layer size (default: $DHID)"
      echo ""
      echo "Subgoal Options:"
      echo "  Trains all subgoals: CleanObject, HeatObject, CoolObject, SliceObject, ToggleObject, PickupObject, PutObject"
      exit 0;;
    *)
      echo "[WARN] Unknown option: $1"; shift;;
  esac
done

# ========= fallback for splits path =========
if [[ ! -f "$SPLITS" ]]; then
  if [[ -f "${DRIVE_ROOT}/data/splits/oct21.json" ]]; then
    SPLITS="${DRIVE_ROOT}/data/splits/oct21.json"
    echo "[INFO] Using Google Drive splits: $SPLITS"
  else
    echo "[WARN] Splits file not found: $SPLITS"
    echo "[INFO] Will use default splits from Hugging Face dataset"
  fi
fi

# ========= environment setup =========
export MCR_ROOT="$MCR_ROOT"
mkdir -p "$DOUT"
cd "$MCR_ROOT/Interactions"
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/Interactions:${PYTHONPATH}"

# ========= check and install dependencies for streaming =========
if [[ "$USE_STREAMING" -eq 1 ]]; then
  echo "[INFO] Checking required packages for streaming..."
  if ! python -c "import datasets, huggingface_hub" 2>/dev/null; then
    echo "[INFO] Installing streaming dependencies..."
    pip install datasets huggingface-hub requests
  fi
fi

# ========= subgoal training loop =========
subgoals=("CleanObject" "HeatObject" "CoolObject" "SliceObject" "ToggleObject" "PickupObject" "PutObject")

for subgoal in "${subgoals[@]}"; do
  echo "===================================================================="
  echo "[INFO] Training subgoal: $subgoal"
  echo "===================================================================="

  # Create output directory for this subgoal
  SUBGOAL_DOUT="${DOUT}/${subgoal}"
  mkdir -p "$SUBGOAL_DOUT"

  # Assemble training command
  CMD=( python models/train/train_seq2seq_stream.py
    --seed "$SEED"
    --huggingface_id "$HUGGINGFACE_ID"
    --splits "$SPLITS"
    --pp_folder "$PP_FOLDER"
    --model "$MODEL"
    --dout "$SUBGOAL_DOUT"
    --subgoal_analysis "$subgoal"
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

  # Add optional flags
  [[ "$SAVE_EVERY_EPOCH" -eq 1 ]] && CMD+=( --save_every_epoch )
  [[ "$USE_STREAMING" -eq 1 ]] && CMD+=( --use_streaming )
  [[ "$GPU" -eq 1 ]] && CMD+=( --gpu )
  [[ -n "$RESUME" ]] && CMD+=( --resume "$RESUME" )
  [[ "$USE_TEMPLATED_GOALS" -eq 1 ]] && CMD+=( --use_templated_goals )
  [[ "$ZERO_GOAL" -eq 1 ]] && CMD+=( --zero_goal )
  [[ "$ZERO_INSTR" -eq 1 ]] && CMD+=( --zero_instr )
  [[ "$DEC_TEACHER_FORCING" -eq 1 ]] && CMD+=( --dec_teacher_forcing )
  [[ "$TEMP_NO_HISTORY" -eq 1 ]] && CMD+=( --temp_no_history )
  [[ "$PANORAMIC" -eq 1 ]] && CMD+=( --panoramic )
  [[ "$ORIENTATION" -eq 1 ]] && CMD+=( --orientation )
  [[ "$PANORAMIC_CONCAT" -eq 1 ]] && CMD+=( --panoramic_concat )
  [[ "$FAST_EPOCH" -eq 1 ]] && CMD+=( --fast_epoch )
  [[ "$DATASET_FRACTION" -gt 0 ]] && CMD+=( --dataset_fraction "$DATASET_FRACTION" )

  echo "[INFO] MCR_ROOT  = $MCR_ROOT"
  echo "[INFO] DOUT      = $SUBGOAL_DOUT"
  echo "[INFO] MODEL     = $MODEL"
  echo "[INFO] HF_ID     = $HUGGINGFACE_ID"
  echo "[INFO] STREAMING = $USE_STREAMING"
  echo "[INFO] BATCH     = $BATCH"
  echo "[INFO] EPOCH     = $EPOCH"
  echo "[INFO] CMD: ${CMD[*]}"

  # Execute training command
  echo "[INFO] Starting training for $subgoal..."
  start_time=$(date +%s)

  # Run the command with error handling
  if ! "${CMD[@]}"; then
    echo "[ERROR] Training failed for subgoal: $subgoal"
    # Continue with next subgoal instead of exiting
    continue
  fi

  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "[INFO] Completed $subgoal in $((duration / 60))m $((duration % 60))s"

done

echo "===================================================================="
echo "[INFO] All subgoal training completed!"
echo "[INFO] Models saved to: $DOUT"
echo "===================================================================="

# ========= create summary file =========
SUMMARY_FILE="${DOUT}/training_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Training Summary
================
Date: $(date)
MCR_ROOT: $MCR_ROOT
HuggingFace ID: $HUGGINGFACE_ID
Streaming Mode: $USE_STREAMING

Hyperparameters:
- Batch size: $BATCH
- Epochs: $EPOCH
- Learning rate: $LR
- Hidden size: $DHID
- Embedding size: $DEMB

Trained Subgoals:
$(printf -- '- %s\n' "${subgoals[@]}")

Output directories:
$(for subgoal in "${subgoals[@]}"; do echo "- ${DOUT}/${subgoal}"; done)
EOF

echo "[INFO] Summary saved to: $SUMMARY_FILE"