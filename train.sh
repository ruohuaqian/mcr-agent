#!/bin/bash

# Define common variables with Google Drive integration
ALFRED_ROOT=/content/alfred
MCR_ROOT=/content/mcr-agent
DRIVE_ROOT=/content/drive/MyDrive

# Use Google Drive for data storage if available
if [ -d "$DRIVE_ROOT/json_feat_2.1.0" ]; then
    DATA_PATH="$DRIVE_ROOT/json_feat_2.1.0"
    echo "Using Google Drive dataset: $DATA_PATH"
else
    DATA_PATH="$ALFRED_ROOT/data/json_feat_2.1.0"
    echo "Using local dataset: $DATA_PATH"
fi

# Set output to Google Drive
DOUT="${DOUT:-$DRIVE_ROOT/exp/MCR_Agent}"
BATCH_SIZE=8  # 减少默认批处理大小以避免内存不足
PANORAMIC_ARGS="--panoramic --panoramic_concat"

# 初始化其他参数
SPLITS=""
LR=""
PREPROCESS=""
DATASET_FRACTION=""
FAST_EPOCH=""
DROPOUT_ARGS=""
MODEL_ARGS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --dout)
            DOUT="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --preprocess)
            PREPROCESS="--preprocess"
            shift
            ;;
        # 添加内存优化参数
        --dataset_fraction)
            DATASET_FRACTION="--dataset_fraction $2"
            shift 2
            ;;
        --fast_epoch)
            FAST_EPOCH="--fast_epoch"
            shift
            ;;
        --vis_dropout)
            DROPOUT_ARGS="$DROPOUT_ARGS --vis_dropout $2"
            shift 2
            ;;
        --hstate_dropout)
            DROPOUT_ARGS="$DROPOUT_ARGS --hstate_dropout $2"
            shift 2
            ;;
        --dhid)
            MODEL_ARGS="$MODEL_ARGS --dhid $2"
            shift 2
            ;;
        --demb)
            MODEL_ARGS="$MODEL_ARGS --demb $2"
            shift 2
            ;;
        --amp)
            # 虽然参数列表中没有，但可以尝试传递
            MODEL_ARGS="$MODEL_ARGS --amp"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

# 构建通用参数
COMMON_ARGS="--batch ${BATCH_SIZE} --gpu --save_every_epoch --data ${DATA_PATH}"
if [ -n "$SPLITS" ]; then
    COMMON_ARGS="$COMMON_ARGS --splits $SPLITS"
fi
if [ -n "$PREPROCESS" ]; then
    COMMON_ARGS="$COMMON_ARGS $PREPROCESS"
fi
if [ -n "$DATASET_FRACTION" ]; then
    COMMON_ARGS="$COMMON_ARGS $DATASET_FRACTION"
fi
if [ -n "$FAST_EPOCH" ]; then
    COMMON_ARGS="$COMMON_ARGS $FAST_EPOCH"
fi
if [ -n "$DROPOUT_ARGS" ]; then
    COMMON_ARGS="$COMMON_ARGS $DROPOUT_ARGS"
fi
if [ -n "$MODEL_ARGS" ]; then
    COMMON_ARGS="$COMMON_ARGS $MODEL_ARGS"
fi

# Create necessary directories
mkdir -p "$DOUT"
mkdir -p "$MCR_ROOT/exp"

# Create symbolic links to Google Drive outputs
ln -sf "$DOUT" "$MCR_ROOT/exp/MCR_Agent"

# Navigate to the MCR-Agent root and set up common symbolic links
cd ${MCR_ROOT}

# Create environment links for all components
components=("MasterPolicy" "Interactions" "PCC" "OEM")
for comp in "${components[@]}"; do
    mkdir -p "${MCR_ROOT}/${comp}/env"
    mkdir -p "${MCR_ROOT}/${comp}/gen"
    mkdir -p "${MCR_ROOT}/${comp}/exp"

    # Link to main directories
    ln -sf ${MCR_ROOT}/env ${MCR_ROOT}/${comp}/env
    ln -sf ${MCR_ROOT}/gen ${MCR_ROOT}/${comp}/gen
    ln -sf ${MCR_ROOT}/exp ${MCR_ROOT}/${comp}/exp
done

echo "Starting MCR-Agent training with output directory: $DOUT"
echo "Common arguments: $COMMON_ARGS"

################################## Train master policy ##################################
echo "=== Training Master Policy ==="
cd ${MCR_ROOT}/MasterPolicy
python models/train/train_seq2seq.py \
    --dout ${DOUT}/MasterPolicy \
    --lr ${LR:-1e-4} \
    ${PANORAMIC_ARGS} \
    ${COMMON_ARGS}

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Master Policy training completed successfully"
else
    echo "Master Policy training failed"
    exit 1
fi

cd ${MCR_ROOT}

############################# Training interaction policies #############################
echo "=== Training Interaction Policies ==="
cd ${MCR_ROOT}/Interactions
subgoals=("CleanObject" "HeatObject" "CoolObject" "SliceObject" "ToggleObject" "PickupObject" "PutObject")

for subgoal in "${subgoals[@]}"; do
    echo "Training $subgoal..."
    python models/train/train_seq2seq.py \
        --subgoal_analysis=${subgoal} \
        --dout ${DOUT}/Interactions/${subgoal} \
        --lr ${LR:-1e-3} \
        ${PANORAMIC_ARGS} \
        ${COMMON_ARGS}

    if [ $? -eq 0 ]; then
        echo "$subgoal training completed"
    else
        echo "$subgoal training failed"
    fi
done

cd ${MCR_ROOT}

########################## Train policy composition controller ##########################
echo "=== Training Policy Composition Controller ==="
cd ${MCR_ROOT}/PCC
python models/train/train_seq2seq.py \
    --dout ${DOUT}/PCC \
    --lr ${LR:-1e-3} \
    ${PANORAMIC_ARGS} \
    ${COMMON_ARGS}

if [ $? -eq 0 ]; then
    echo "PCC training completed successfully"
else
    echo "PCC training failed"
fi

cd ${MCR_ROOT}

############################## Train Object Encoding Module #############################
echo "=== Training Object Encoding Module ==="
cd ${MCR_ROOT}/OEM
python models/train/train_seq2seq.py \
    --dout ${DOUT}/OEM \
    --lr ${LR:-1e-3} \
    ${PANORAMIC_ARGS} \
    ${COMMON_ARGS}

if [ $? -eq 0 ]; then
    echo "OEM training completed successfully"
else
    echo "OEM training failed"
fi

cd ${MCR_ROOT}

echo "=== All MCR-Agent components training completed ==="
echo "Output saved to: $DOUT"