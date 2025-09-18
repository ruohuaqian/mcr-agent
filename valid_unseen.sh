#!/usr/bin/env bash
set -euo pipefail

# === 必须先设置你的 MCR 根目录 ===
: "${MCR_ROOT:?Please export MCR_ROOT to the root of mcr-agent}"

# === 只需改这里：OEM 模型路径 ===
OEM_PATH="/content/drive/MyDrive/mcr-agent/exp/OEM/latest.pth"

# === 不常改的固定配置 ===
NAV_PATH="/content/drive/MyDrive/mcr-agent/exp/MasterPolicy/latest.pth"
PICKUP_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/PickupObject/latest.pth"
PUT_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/PutObject/latest.pth"
HEAT_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/HeatObject/latest.pth"
COOL_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/CoolObject/latest.pth"
CLEAN_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/CleanObject/latest.pth"
TOGGLE_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/ToggleObject/latest.pth"
SLICE_PATH="/content/drive/MyDrive/mcr-agent/exp/Interactions/SliceObject/latest.pth"
PCC_PATH="/content/drive/MyDrive/mcr-agent/exp/PCC/latest.pth"

SPLITS="/content/drive/MyDrive/mcr-agent/splits/rest_train.json"
REWARD_CFG="${MCR_ROOT}/models/config/rewards.json"
HF_ID="byeonghwikim/abp_dataset"
EVAL_SPLIT="valid_seen"
MAX_STEP=100
MAX_FAIL=10
NUM_THREADS=4

# === 环境准备 ===
export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/models:${PYTHONPATH:-}"
cd "${MCR_ROOT}/models"

# === 执行 ===
python eval/eval_seq2seq.py \
    --nav_model_path "$NAV_PATH" \
    --pickup_model_path "$PICKUP_PATH" \
    --put_model_path "$PUT_PATH" \
    --heat_model_path "$HEAT_PATH" \
    --cool_model_path "$COOL_PATH" \
    --clean_model_path "$CLEAN_PATH" \
    --toggle_model_path "$TOGGLE_PATH" \
    --slice_model_path "$SLICE_PATH" \
    --object_model_path "$OEM_PATH" \        # 👈 只需要改这一行
    --subgoal_model_path "$PCC_PATH" \
    --splits "$SPLITS" \
    --eval_split "$EVAL_SPLIT" \
    --huggingface_id "$HF_ID" \
    --gpu \
    --max_step "$MAX_STEP" \
    --max_fail "$MAX_FAIL" \
    --reward_config "$REWARD_CFG" \
    --num_threads "$NUM_THREADS"
