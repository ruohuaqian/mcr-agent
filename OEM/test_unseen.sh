#!/usr/bin/env bash
set -euo pipefail

# ---------- 参数解析 ----------
DOUT=""
SPLITS=""
DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dout)   DOUT="$2"; shift 2;;
    --splits) SPLITS="$2"; shift 2;;
    --data)   DATA="$2"; shift 2;;
    --preprocess) PREPROCESS=1; shift;;
    --help|-h)
      echo "Usage: $0 --dout EXP_DIR --splits SPLITS_JSON --data DATA_DIR"
      exit 0;;
    *)
      echo "[WARN] Unknown option: $1"; shift;;
  esac
done

# ---------- 必要性检查 ----------
if [[ -z "${DOUT}" || -z "${SPLITS}" || -z "${DATA}" ]]; then
  echo "[FATAL] --dout / --splits / --data 都必须提供"
  exit 1
fi
if [[ ! -f "$SPLITS" ]]; then
  echo "[FATAL] splits 文件不存在: $SPLITS"
  exit 1
fi
if [[ ! -d "$DATA" ]]; then
  echo "[FATAL] data 目录不存在: $DATA"
  exit 1
fi

# ---------- 环境 ----------
export MCR_ROOT="${MCR_ROOT:-/content/mcr-agent}"
export PYTHONPATH="$MCR_ROOT:$MCR_ROOT/OEM:$PYTHONPATH"
mkdir -p "$DOUT" "$MCR_ROOT/exp"
ln -sfn "$DOUT" "$MCR_ROOT/exp/OEM"

cd "$MCR_ROOT/OEM"
echo "[INFO] MCR_ROOT=$MCR_ROOT"
echo "[INFO] DOUT=$DOUT"
echo "[INFO] SPLITS=$SPLITS"
echo "[INFO] DATA=$DATA"
echo "[INFO] CWD=$(pwd)"

# ---------- 循环评估每个 epoch ----------
for i in $(seq 0 30); do
  CKPT="$DOUT/net_epoch_${i}.pth"
  if [[ ! -f "$CKPT" ]]; then
    echo "[SKIP] 不存在: $CKPT"
    continue
  fi

  echo "[RUN] epoch $i -> $CKPT"
  CUDA_VISIBLE_DEVICES=0 python models/eval/eval_seq2seq.py \
    --model_path "$CKPT" \
    --eval_split valid_unseen \
    --model OEM.models.model.seq2seq_im_mask \
    --data "$DATA" \
    --splits "$SPLITS" \
    --gpu \
    --num_threads 1 \
    --max_steps 400 \
    --max_fails 10 \
    --fast_epoch \
    --preprocess
done
