#!/usr/bin/env bash
set -euo pipefail

# ===================== Defaults =====================
MCR_ROOT=${MCR_ROOT:-/content/mcr-agent}
DRIVE_ROOT=${DRIVE_ROOT:-/content/drive/MyDrive}

DATA_DEFAULT_JSON=${DRIVE_ROOT}/json_feat_2.1.0
SPLITS_DEFAULT_JSON=${DRIVE_ROOT}/mcr-agent/data/splits/oct21.json
DOUT_DEFAULT=${DRIVE_ROOT}/mcr-agent/exp/MasterPolicy

# ---- args aligned with parser ----
SEED=123
DATA="$DATA_DEFAULT_JSON"
SPLITS="$SPLITS_DEFAULT_JSON"
PREPROCESS=0
PP_FOLDER=pp
SAVE_EVERY_EPOCH=0
MODEL=seq2seq_im_mask
GPU=1
DOUT="$DOUT_DEFAULT"
RESUME=""
USE_TEMPLATED_GOALS=0

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
    --data) DATA="$2"; shift 2;;
    --splits) SPLITS="$2"; shift 2;;
    --preprocess) PREPROCESS=1; shift;;
    --pp_folder) PP_FOLDER="$2"; shift 2;;
    --save_every_epoch) SAVE_EVERY_EPOCH=1; shift;;
    --model) MODEL="$2"; shift 2;;
    --gpu) GPU=1; shift;;
    --no-gpu) GPU=0; shift;;
    --dout) DOUT="$2"; shift 2;;
    --resume) RESUME="$2"; shift 2;;
    --use_templated_goals) USE_TEMPLATED_GOALS=1; shift;;

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
      echo "Usage: $0 [--data PATH] [--splits PATH] [--dout PATH] [--model NAME] [--gpu|--no-gpu] ..."
      exit 0;;
    *) echo "[WARN] Unknown option: $1"; shift;;
  esac
done

# ===================== Paths & env =====================
if [[ ! -d "$DATA" ]]; then
  if [[ -d "${DRIVE_ROOT}/json_feat_2.1.0" ]]; then
    DATA="${DRIVE_ROOT}/json_feat_2.1.0"
    echo "[INFO] Use Google Drive dataset: $DATA"
  else
    DATA="$DATA_DEFAULT_JSON"
    echo "[INFO] Use local dataset: $DATA"
  fi
fi

mkdir -p "$DOUT"
mkdir -p "${MCR_ROOT}/exp"
ln -sfn "$DOUT" "${MCR_ROOT}/exp/MasterPolicy"

export MCR_ROOT="$MCR_ROOT"
# 关键点：指向 MasterPolicy 子项目，避免错误引用其他模块实现
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/MasterPolicy:${PYTHONPATH}"

cd "${MCR_ROOT}/MasterPolicy"

# ===================== Build cmd =====================
CMD=( python models/train/train_seq2seq.py
  --seed "$SEED"
  --data "$DATA"
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

# flags
[[ "$PREPROCESS" -eq 1 ]] && CMD+=( --preprocess )
[[ "$SAVE_EVERY_EPOCH" -eq 1 ]] && CMD+=( --save_every_epoch )
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

echo "[INFO] MCR_ROOT    = $MCR_ROOT"
echo "[INFO] DOUT        = $DOUT"
echo "[INFO] MODEL       = $MODEL"
echo "[INFO] CMD: ${CMD[*]}"
"${CMD[@]}"
