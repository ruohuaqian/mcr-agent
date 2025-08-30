#!/bin/bash

# Define common variables to avoid repetition
ALFRED_ROOT=/content/alfred
MCR_ROOT=/content/mcr-agent
DATA_PATH=$ALFRED_ROOT/data/json_feat_2.1.0
BATCH_SIZE=16
PANORAMIC_ARGS="--panoramic --panoramic_concat"
COMMON_ARGS="--batch ${BATCH_SIZE} --gpu --save_every_epoch --preprocess --data ${DATA_PATH}"
#!/bin/bash

# Define common variables
ALFRED_ROOT=/content/alfred
MCR_ROOT=/content/mcr-agent
DATA_PATH=$ALFRED_ROOT/data/json_feat_2.1.0
BATCH_SIZE=16
PANORAMIC_ARGS="--panoramic --panoramic_concat"
COMMON_ARGS="--batch ${BATCH_SIZE} --gpu --save_every_epoch --preprocess --data ${DATA_PATH}"

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
        *)
            shift
            ;;
    esac
done

# Navigate to the MCR-Agent root and set up common symbolic links
cd ${MCR_ROOT}
export ALFRED_ROOT=${ALFRED_ROOT}
ln -s ${MCR_ROOT}/env ${MCR_ROOT}/MasterPolicy/env
ln -s ${MCR_ROOT}/gen ${MCR_ROOT}/MasterPolicy/gen
ln -s ${MCR_ROOT}/exp ${MCR_ROOT}/MasterPolicy/exp

ln -s ${MCR_ROOT}/env ${MCR_ROOT}/Interactions/env
ln -s ${MCR_ROOT}/gen ${MCR_ROOT}/Interactions/gen
ln -s ${MCR_ROOT}/exp ${MCR_ROOT}/Interactions/exp

ln -s ${MCR_ROOT}/env ${MCR_ROOT}/PCC/env
ln -s ${MCR_ROOT}/gen ${MCR_ROOT}/PCC/gen
ln -s ${MCR_ROOT}/exp ${MCR_ROOT}/PCC/exp

ln -s ${MCR_ROOT}/env ${MCR_ROOT}/OEM/env
ln -s ${MCR_ROOT}/gen ${MCR_ROOT}/OEM/gen
ln -s ${MCR_ROOT}/exp ${MCR_ROOT}/OEM/exp

# The original script links were being created from the subdirectory, which is incorrect.
# The corrected script creates all links once from the main MCR_ROOT.

################################## Train master policy ##################################
cd ${MCR_ROOT}/MasterPolicy
python models/train/train_seq2seq.py \
    --dout ${DOUT:-${MCR_ROOT}/exp/MasterPolicy} \
    --lr ${LR:-1e-4} \
    ${PANORAMIC_ARGS} \
    --batch ${BATCH_SIZE} \
    --gpu \
    --save_every_epoch \
    ${PREPROCESS:-} \
    --data ${DATA_PATH} \
    ${SPLITS:+--splits ${SPLITS}} \

cd ${MCR_ROOT}

############################# Training interaction policies #############################
cd ${MCR_ROOT}/Interactions
subgoals=(CleanObject HeatObject CoolObject SliceObject ToggleObject PickupObject PutObject)
for subgoal in "${subgoals[@]}"; do
    python models/train/train_seq2seq.py \
        --subgoal_analysis=${subgoal} \
        --dout ${MCR_ROOT}/exp/${subgoal} \
        --lr 1e-3 \
        ${COMMON_ARGS}
done
cd ${MCR_ROOT}

########################## Train policy composition controller ##########################
cd ${MCR_ROOT}/PCC
python models/train/train_seq2seq.py \
    --dout ${MCR_ROOT}/exp/PCC \
    --lr 1e-3 \
    ${PANORAMIC_ARGS} \
    ${COMMON_ARGS}

cd ${MCR_ROOT}

############################## Train Object Encoding Module #############################
cd ${MCR_ROOT}/OEM
python models/train/train_seq2seq.py \
    --dout ${MCR_ROOT}/exp/OEM \
    --lr 1e-3 \
    ${PANORAMIC_ARGS} \
    ${COMMON_ARGS}

cd ${MCR_ROOT}