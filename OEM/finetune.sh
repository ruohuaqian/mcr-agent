#!/usr/bin/env bash
set -euo pipefail
MCR_ROOT=${MCR_ROOT:-/content/mcr-agent}
# Defaults
CKPT=""
OUT=""
MODEL="OEM.models.model.seq2seq_im_mask_obj"
SPLITS="data/splits/oct21.json"
HF_ID="byeonghwikim/abp_dataset"
PP_FOLDER="pp"
BATCH=128
EPOCHS=10
LR=1e-4
SEED=123
FREEZE=""
GPU=1
USE_STREAMING=1
EXTRA_ARGS=""

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") --ckpt PATH --out DIR [options]

Required:
  --ckpt PATH              Path to init checkpoint (only weights are used)
  --out DIR                Output dir for this fine-tune run

Common options:
  --model STR              Python module path to model (default: $MODEL)
  --splits PATH            Split json (default: $SPLITS)
  --hf-id STR              HF dataset id (default: $HF_ID)
  --pp-folder STR          Preprocess folder name (default: $PP_FOLDER)
  --batch INT              Batch size (default: $BATCH)
  --epochs INT             Epochs (default: $EPOCHS)
  --lr FLOAT               Learning rate (default: $LR)
  --seed INT               Random seed (default: $SEED)
  --freeze STR             Comma-separated prefixes to freeze (e.g. "encoder,visual_backbone,embeddings")
  --gpu 0|1                Use GPU flag (default: $GPU)
  --streaming 0|1          Use HF streaming (default: $USE_STREAMING)
  --extra "ARGS"           Extra args passed to python (quoted)

Example:
  ./finetune.sh \\
    --ckpt /content/drive/MyDrive/mcr-agent/exp/pretrain/last.pth \\
    --out  /content/drive/MyDrive/mcr-agent/exp/OEM_ft_001 \\
    --model OEM.models.model.seq2seq_im_mask_obj \\
    --splits /content/drive/MyDrive/mcr-agent/splits/rest_train.json \\
    --batch 256 --epochs 10 --lr 1e-4 \\
    --freeze "encoder,visual_backbone,embeddings"

USAGE
  exit 1
}

# Parse args (long-only)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --splits) SPLITS="$2"; shift 2;;
    --hf-id) HF_ID="$2"; shift 2;;
    --pp-folder) PP_FOLDER="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --freeze) FREEZE="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --streaming) USE_STREAMING="$2"; shift 2;;
    --extra) EXTRA_ARGS="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -z "$CKPT" || -z "$OUT" ]] && usage

mkdir -p "$OUT"
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/OEM:${PYTHONPATH}"
cd "${MCR_ROOT}/OEM"
# Build flags
GPU_FLAG=$([ "$GPU" = "1" ] && echo "--gpu" || echo "")
STREAM_FLAG=$([ "$USE_STREAMING" = "1" ] && echo "--use_streaming" || echo "")
FREEZE_FLAG=$([ -n "$FREEZE" ] && echo "--freeze \"$FREEZE\"" || echo "")

CMD="python -u train_seq2seq_stream.py \
  --seed $SEED \
  --huggingface_id $HF_ID \
  --splits $SPLITS \
  --pp_folder $PP_FOLDER \
  --model $MODEL \
  $GPU_FLAG \
  --dout $OUT \
  --finetune_from $CKPT \
  $STREAM_FLAG \
  --batch $BATCH \
  --epoch $EPOCHS \
  --lr $LR \
  $FREEZE_FLAG \
  $EXTRA_ARGS"

echo "[finetune] Launching:"
echo "$CMD"
echo "$CMD" > "$OUT/cmd.txt"

# shellcheck disable=SC2086
eval $CMD
