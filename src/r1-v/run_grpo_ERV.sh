#!/bin/bash

cd src/r1-v
SRC_DIR=/mnt/ssd_hs/Exp/R1-Omni
CURRENT_TIME=$(TZ='Asia/Seoul' date +"%m-%d-%H%M")

export OMP_NUM_THREADS=8
export PYTHONPATH=${SRC_DIR}/src/r1-v/src:$PYTHONPATH
export PYTHONPATH=${SRC_DIR}:$PYTHONPATH
export WANDB_API_KEY=c0a63e8ee4d91080b2a552e5d9afb873d303801d
export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/$CURRENT_TIME-grpo.txt"
export HF_HOME=${SRC_DIR}/hf_cache
export HF_HUB_CACHE=${SRC_DIR}/hf_cache


mkdir -p ./logs
if [[ "$LOCAL_RANK" == "" || "$LOCAL_RANK" == "0" ]]; then
    exec > >(tee -a "$LOG_PATH") 2>&1
else
    exec > /dev/null 2>&1
fi
echo "Log path set to: $LOG_PATH"


# Configuration
model_size="0.5B" # 0.5B / 7B
if [ "$model_size" == "0.5B" ]; then
    num_generations=4
    vision_tower_path="${SRC_DIR}/pre-trained/siglip-base-patch16-224"
elif [ "$model_size" == "7B" ]; then
    num_generations=4
    vision_tower_path="${SRC_DIR}/pre-trained/siglip-so400m-patch14-384"
fi
audio_tower_path="${SRC_DIR}/pre-trained/whisper-large-v3"
bert_model_path="${SRC_DIR}/pre-trained/bert-base-uncased"


# Training Hyperparameters
train_epoch=2
data_ratio=1.0
eval_strategy=no # no steps epoch
eval_steps=1
batch_size=1
grad_accum=2
LR=1e-6
beta=0.04

# Pre-trained SFT path
# MODEL_NAME=${SRC_DIR}/pre-trained_sft_path
MODEL_NAME=Rhatanii/ERV-0.5B #${SRC_DIR}/pre-trained/HumanOmni-0.5B

# Set Save Strategy & Name
MODEL_ABBR="EMER-SFT"
REWARD_LIST=("format" "accuracy" "think_step")
REWARD_ABBR=$(
  for item in "${REWARD_LIST[@]}"; do
    IFS='_' read -ra parts <<< "$item"
    abbr=""
    for p in "${parts[@]}"; do
      abbr+="${p:0:1}"
    done
    printf "%s_" "$abbr"
  done | sed 's/_$//'
)

echo "$REWARD_ABBR"
SAVE_STRATEGY="epoch"

RUN_NAME=results-${CURRENT_TIME}-${MODEL_ABBR}-${REWARD_ABBR}-ep${train_epoch}-ratio${data_ratio}-G${num_generations}-lr${LR}-bs${batch_size}-ga${grad_accum}-b${beta}
OUTPUT_DIR=${SRC_DIR}/src/r1-v/${RUN_NAME}


# Dataset paths
DATASET_NAME=${SRC_DIR}/pre-trained/dataset_file/RLVR.json
TEST_DATASET_NAME=${SRC_DIR}/pre-trained/dataset_file/RLVR.json


WANDB_MODE=online torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    ${SRC_DIR}/src/r1-v/src/open_r1/grpo_ERV.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME} \
    --bert_model_path ${bert_model_path} \
    --vision_tower_path ${vision_tower_path} \
    --audio_tower_path ${audio_tower_path} \
    --dataset_name ${DATASET_NAME} \
    --test_dataset_name ${TEST_DATASET_NAME} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 256 \
    --max_completion_length 512 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs ${train_epoch} \
    --run_name ${RUN_NAME} \
    --save_strategy=${SAVE_STRATEGY} \
    --save_only_model false \
    --num_generations ${num_generations} \
    --reward_funcs "${REWARD_LIST[@]}" \
    --reference_model_switch true \
    --data_ratio ${data_ratio} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --learning_rate ${LR} \
    --dataloader_num_workers 8 \
    --beta ${beta} 



