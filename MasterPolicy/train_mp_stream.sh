#!/usr/bin/env bash
set -euo pipefail

# ===================== Defaults =====================
MCR_ROOT=${MCR_ROOT:-/content/mcr-agent}
DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/MyDrive}

SPLITS_DEFAULT_JSON=${DRIVE_ROOT}/mcr-agent/data/splits/oct21.json
DOUT_DEFAULT=${DRIVE_ROOT}/mcr-agent/exp/MasterPolicy

# ---- args aligned with parser ----
SEED=123
HUGGINGFACE_ID="byeonghwikim/abp_dataset"  # 默认Hugging Face数据集ID
SPLITS="$SPLITS_DEFAULT_JSON"
PP_FOLDER="pp"
SAVE_EVERY_EPOCH=0
MODEL="MasterPolicy.models.model.seq2seq_im_mask"
GPU=1
DOUT="$DOUT_DEFAULT"
RESUME=""
USE_TEMPLATED_GOALS=0
USE_STREAMING=1  # 默认启用流式模式

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
PANORAMIC=0
ORIENTATION=0
PANORAMIC_CONCAT=0

FAST_EPOCH=0
DATASET_FRACTION=0

# ===================== CLI parse =====================
while [[ $# -gt 0 ]]; do
  case "$1" in
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
      echo "Streaming Mode Options:"
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
      echo "Other options same as before"
      exit 0;;
    *) echo "[WARN] Unknown option: $1"; shift;;
  esac
done

# ===================== Paths & env =====================
# 检查splits文件是否存在
if [[ ! -f "$SPLITS" ]]; then
  if [[ -f "${DRIVE_ROOT}/mcr-agent/data/splits/oct21.json" ]]; then
    SPLITS="${DRIVE_ROOT}/mcr-agent/data/splits/oct21.json"
    echo "[INFO] Using Google Drive splits: $SPLITS"
  else
    echo "[WARN] Splits file not found: $SPLITS"
    echo "[INFO] Will attempt to use splits from Hugging Face dataset"
  fi
fi

mkdir -p "$DOUT"
export MCR_ROOT="$MCR_ROOT"
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/MasterPolicy:${PYTHONPATH}"

cd "${MCR_ROOT}/MasterPolicy"

# ===================== Check streaming dependencies =====================
if [[ "$USE_STREAMING" -eq 1 ]]; then
  echo "[INFO] Checking required packages for streaming..."
  if ! python -c "import datasets, huggingface_hub, requests" 2>/dev/null; then
    echo "[INFO] Installing streaming dependencies..."
    pip install datasets huggingface-hub requests
  fi
fi

# ===================== Build cmd =====================
# 使用流式训练脚本
TRAIN_SCRIPT="models/train/train_seq2seq_stream.py"

CMD=( python "$TRAIN_SCRIPT"
  --seed "$SEED"
  --huggingface_id "$HUGGINGFACE_ID"
  --splits "$SPLITS"
  --pp_folder "$PP_FOLDER"
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

# 添加可选标志
[[ "$SAVE_EVERY_EPOCH" -eq 1 ]] && CMD+=( --save_every_epoch )
[[ "$GPU" -eq 1 ]] && CMD+=( --gpu )
[[ -n "$RESUME" ]] && CMD+=( --resume "$RESUME" )
[[ "$USE_TEMPLATED_GOals" -eq 1 ]] && CMD+=( --use_templated_goals )
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

# ===================== Execute =====================
echo "[INFO] MCR_ROOT    = $MCR_ROOT"
echo "[INFO] DOUT        = $DOUT"
echo "[INFO] MODEL       = $MODEL"
echo "[INFO] HF_ID       = $HUGGINGFACE_ID"
echo "[INFO] STREAMING   = $USE_STREAMING"
echo "[INFO] BATCH       = $BATCH"
echo "[INFO] EPOCH       = $EPOCH"
echo "[INFO] CMD: ${CMD[*]}"

# 执行训练命令
echo "[INFO] Starting training..."
start_time=$(date +%s)

if ! "${CMD[@]}"; then
    echo "[ERROR] Training failed with exit code $?"
    exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "[INFO] Training completed successfully in $((duration / 60))m $((duration % 60))s"
echo "[INFO] Model saved to: $DOUT"

# ===================== Create summary =====================
SUMMARY_FILE="${DOUT}/training_summary.txt"
cat > "$SUMMARY_FILE" << EOF
Training Summary
================
Date: $(date)
Model: $MODEL
HuggingFace Dataset: $HUGGINGFACE_ID
Streaming Mode: $USE_STREAMING

Hyperparameters:
- Seed: $SEED
- Batch size: $BATCH
- Epochs: $EPOCH
- Learning rate: $LR
- Hidden size: $DHID
- Embedding size: $DEMB

Loss Weights:
- Mask loss: $MASK_LOSS_WT
- Action loss: $ACTION_LOSS_WT
- Subgoal aux: $SUBGOAL_AUX_LOSS_WT
- PM aux: $PM_AUX_LOSS_WT

Dropouts:
- Lang: $LANG_DROPOUT
- Input: $INPUT_DROPOUT
- Visual: $VIS_DROPOUT
- Hidden state: $HSTATE_DROPOUT
- Attention: $ATTN_DROPOUT
- Actor: $ACTOR_DROPOUT

Features:
- Zero goal: $ZERO_GOAL
- Zero instr: $ZERO_INSTR
- Panoramic: $PANORAMIC
- Orientation: $ORIENTATION
- Panoramic concat: $PANORAMIC_CONCAT

Training Time: $((duration / 60)) minutes $((duration % 60)) seconds
EOF

echo "[INFO] Training summary saved to: $SUMMARY_FILE"