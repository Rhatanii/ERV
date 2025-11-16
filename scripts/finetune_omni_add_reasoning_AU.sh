#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-6}
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
GLOBAL_BATCH_SIZE=36
LOCAL_BATCH_SIZE=2


GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo $GRADIENT_ACCUMULATION_STEPS
SAVE_STEPS=50
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=humanomniqwen2_siglip
export HF_HOME=/mnt/ssd_hs/.cache
export HF_HUB_CACHE=/mnt/ssd_hs/.cache
# export HF_ENDPOINT=http://hf-mirror.com

# TRAIN_EPOCHS=$3
# LR=$4

LR=2e-5

# /mnt/ssd_hs/Dataset/R1-Omni/cold_start/Unified/EMER_MAFW_DFEW-ERV-AU_cold_start.json
DATASET_NAME=EMER_MAFW_DFEW-ERV_cold_start_new_format_av #merr-fine_mer-caption-plus
CONFIG_NAME=${DATASET_NAME}.yaml
CURRENT_TIME=$(TZ='Asia/Seoul' date +"%m-%d-%H%M")
# left_padding=True

for TRAIN_EPOCHS in 5
do
    for MODEL_BASE_NAME in HumanOmni
    do
        TRAIN_CONCEPT=${CURRENT_TIME}-${DATASET_NAME}-bs${GLOBAL_BATCH_SIZE}-ga${GRADIENT_ACCUMULATION_STEPS}
        # MODEL_BASE_NAME=R1-Omni # R1-Omni HumanOmni
        MM_TUNABLE_PARTS="mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower"
        RUN_NAME=${TRAIN_CONCEPT}-${MODEL_BASE_NAME}-EPOCH${TRAIN_EPOCHS}-LR${LR}-${MM_TUNABLE_PARTS}
        OUTP_DIR=work_dirs

        torchrun --nnodes $WORLD_SIZE \
            --nproc_per_node $NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            --node_rank $RANK \
            humanomni/train_flash_attn.py \
            --deepspeed scripts/zero3.json \
            --model_type HumanOmni_qwen2 \
            --model_path /mnt/ssd_hs/Exp/R1-Omni/pre-trained/HumanOmni-0.5B \
            --vision_tower /mnt/ssd_hs/Exp/R1-Omni/pre-trained/siglip-base-patch16-224 \
            --audio_tower /mnt/ssd_hs/Exp/R1-Omni/pre-trained/whisper-large-v3 \
            --mm_projector_type all_in_one_small \
            --mm_tunable_parts ${MM_TUNABLE_PARTS} \
            --data_path   ./configs/train/sft/${CONFIG_NAME} \
            --data_folder / \
            --mm_vision_select_layer -2 \
            --image_aspect_ratio pad \
            --num_frames 8 \
            --bf16 True \
            --tf32 True \
            --fp16 False \
            --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
            --num_train_epochs ${TRAIN_EPOCHS} \
            --per_device_train_batch_size $LOCAL_BATCH_SIZE \
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
            --dataloader_num_workers 8 \
            --report_to wandb \
            --seed 17 \
            --save_strategy "steps" \
            --save_steps ${SAVE_STEPS} \
            # --left_padding ${left_padding} 
    done
done
