#!/usr/bin/env bash
set -euo pipefail

# ========= default =========
MCR_ROOT=${MCR_ROOT:-$(pwd)}
DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/MyDrive}
SPLITS_DEFAULT_JSON=${DRIVE_ROOT}/data/splits/oct21.json

# ========= default hyperparameters =========
SEED=123
HUGGINGFACE_ID="byeonghwikim/abp_dataset"
SPLITS="$SPLITS_DEFAULT_JSON"
PP_FOLDER="pp"
SAVE_EVERY_EPOCH=0
MODEL="PCC.models.model.seq2seq_im_mask"
GPU=1
DOUT="${MCR_ROOT}/exp/model:seq2seq_im_mask"
RESUME=""
USE_TEMPLATED_GOALS=0
USE_STREAMING=1  # 默认启用流式加载

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
PANORAMIC=0
ORIENTATION=0
PANORAMIC_CONCAT=0

FAST_EPOCH=0
DATASET_FRACTION=0

# ========= parse command line arguments =========
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
      echo "Usage: $0 [--huggingface_id ID] [--splits PATH] [--dout PATH] [--model NAME] [--subgoal_analysis NAME] [--use_streaming] ..."
      echo ""
      echo "Streaming Mode Options:"
      echo "  --huggingface_id      Hugging Face dataset ID (default: $HUGGINGFACE_ID)"
      echo "  --use_streaming       Enable Hugging Face streaming mode (default: enabled)"
      echo "  --no-streaming        Disable streaming mode"
      echo ""
      echo "Required:"
      echo ""
      echo "Other options same as original script"
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
cd "$MCR_ROOT/PCC"
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/PCC:${PYTHONPATH}"

# ========= training with streaming =========
echo "[INFO] Training with Hugging Face Streaming Mode"
echo "[INFO] Dataset: $HUGGINGFACE_ID"

# Assemble training command
CMD=( python models/train/train_seq2seq_stream.py  # 使用流式训练脚本
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

# Add optional flags
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

echo "[INFO] MCR_ROOT = $MCR_ROOT"
echo "[INFO] DOUT     = $DOUT"
echo "[INFO] MODEL    = $MODEL"
echo "[INFO] HF_ID    = $HUGGINGFACE_ID"
echo "[INFO] STREAMING= $USE_STREAMING"
echo "[INFO] CMD: ${CMD[*]}"

# 检查是否安装了必要的库
if [[ "$USE_STREAMING" -eq 1 ]]; then
  echo "[INFO] Checking required packages for streaming..."
  pip list | grep -E "datasets|huggingface-hub" || {
    echo "[INFO] Installing streaming dependencies..."
    pip install datasets huggingface-hub
  }
fi

# 执行训练命令
"${CMD[@]}"

echo "[INFO] Training completed successfully!"
echo "[INFO] Model saved to: $DOUT"