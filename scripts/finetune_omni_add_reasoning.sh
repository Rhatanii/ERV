#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16667
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"


# Training Arguments
GLOBAL_BATCH_SIZE=32
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
TRAIN_EPOCHS=10
LR=2e-5
MODEL_SIZE=7B


# Log Arguments
export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT=cold_start_SFT



# Arguments
SRC="Enter your src dir path" 
MODEL_SRC=${SRC_NAME}/pre-trained
CONFIG_NAME=emotion_emer.yaml 
CURRENT_TIME=$(date +"%m-%d-%H%M")
RUN_NAME=${CURRENT_TIME}-Ep${TRAIN_EPOCHS}-lr${LR}
OUT_DIR=checkpoints_SFT



if [ "${MODEL_SIZE}" == "0.5B" ]; then
    MODEL_NAME=${MODEL_SRC}/HumanOmni-0.5B
    MM_PROJ_TYPE="all_in_one_small"
    VISION_TOWER_NAME=${MODEL_SRC}/siglip-base-patch16-224
elif [ "${MODEL_SIZE}" == "7B" ]; then
    MODEL_NAME=${MODEL_SRC}/HumanOmni-7B
    MM_PROJ_TYPE="all_in_one"
    VISION_TOWER_NAME=${MODEL_SRC}/siglip-so400m-patch14-384


# Learnable parts
MM_TUNABLE_PARTS="mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower"


torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    humanomni/train_flash_attn.py \
    --deepspeed scripts/zero3.json \
    --model_type HumanOmni_qwen2 \
    --model_path ${MODEL_NAME} \
    --vision_tower ${VISION_TOWER_NAME} \
    --audio_tower ${MODEL_SRC}/whisper-large-v3 \
    --mm_projector_type ${MM_PROJ_TYPE} \
    --mm_tunable_parts ${MM_TUNABLE_PARTS} \
    --data_path   ./configs/train/sft/${CONFIG_NAME} \
    --data_folder / \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUT_DIR}/${WANDB_PROJECT}/SFT_${RUN_NAME} \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --per_device_train_batch_size ${LOCAL_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_total_limit 99 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --mm_use_x_start_end True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --seed 17 \
    --save_strategy "epoch"
