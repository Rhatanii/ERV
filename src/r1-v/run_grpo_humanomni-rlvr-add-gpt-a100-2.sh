#!/bin/bash
cd src/r1-v

SRC_DIR=/mnt/ssd_hs

export WANDB_API_KEY=c0a63e8ee4d91080b2a552e5d9afb873d303801d
export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=${SRC_DIR}/.cache
export HF_HUB_CACHE=${SRC_DIR}/.cache
mkdir -p ./logs

echo "Log path set to: $LOG_PATH"

train_epoch=2
data_ratio=1.0
eval_strategy=no # no steps epoch
eval_steps=1
num_generations=4
LR=1e-6
MODEL_NAME=${SRC_DIR}/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0515-EMER-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower

# EMER-MERR 0.5B : /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0525-EMER-MERR-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower
#7B:  /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0605-EMER-MERR-8frame-all_in_one-HumanOmni-EPOCH10-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-750


MODEL_ABBR=EMER-SFT-0.5B
REWARD_LIST=("format" "accuracy" "think_step_with_neutral")
REWARD_ABBR=FORMAT_ACC_THINK-STEP-NEUTRAL
# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0601-EMER-MERR-fr32-HumanOmni-EPOCH5-LR1e-4-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-745

CURRENT_TIME=$(date +"%m-%d-%H%M")




RUN_CONCEPT=${CURRENT_TIME}-${MODEL_ABBR}-${REWARD_ABBR}
RUN_NAME=results-${RUN_CONCEPT}-epoch${train_epoch}-ratio${data_ratio}-grpo${num_generations}-lr${LR}
OUTPUT_DIR=${SRC_DIR}/Exp/R1-Omni/src/r1-v/${RUN_NAME}
MODEL_DIR=${MODEL_NAME}
DATA_SRC=${SRC_DIR}/Dataset
DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR.json
TEST_DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-FINE.json


WANDB_MODE=online torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12112" \
    /mnt/ssd_hs/Exp/R1-Omni/src/r1-v/src/open_r1/grpo-text_multi_label.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_DIR} \
    --dataset_name ${DATASET_NAME} \
    --test_dataset_name ${TEST_DATASET_NAME} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs ${train_epoch} \
    --run_name ${RUN_NAME} \
    --save_strategy="epoch" \
    --save_only_model false \
    --num_generations ${num_generations} \
    --reward_funcs "${REWARD_LIST[@]}" \
    --reference_model_switch true \
    --data_ratio ${data_ratio} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --learning_rate ${LR} \
