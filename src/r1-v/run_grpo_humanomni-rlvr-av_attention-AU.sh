#!/bin/bash

cd src/r1-v

SRC_DIR=/mnt/ssd_hs
export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=4
export PYTHONPATH=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v/src:$PYTHONPATH
export PYTHONPATH=/mnt/ssd_hs/Exp/R1-Omni:$PYTHONPATH

CURRENT_TIME=$(TZ='Asia/Seoul' date +"%m-%d-%H%M")


export WANDB_API_KEY=c0a63e8ee4d91080b2a552e5d9afb873d303801d
export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/$CURRENT_TIME-humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=${SRC_DIR}/hf_cache
export HF_HUB_CACHE=${SRC_DIR}/hf_cache


mkdir -p ./logs

if [[ "$LOCAL_RANK" == "" || "$LOCAL_RANK" == "0" ]]; then
    exec > >(tee -a "$LOG_PATH") 2>&1
else
    exec > /dev/null 2>&1
fi


echo "Log path set to: $LOG_PATH"

train_epoch=2
data_ratio=1.0
eval_strategy=no # no steps epoch
eval_steps=1
num_generations=16
batch_size=2
grad_accum=2
LR=1e-6
beta=0.04
select_layer_idx=mean # mean

MODEL_NAME=${SRC_DIR}/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_11-06-2025-EMER_MAFW_DFEW-ERV_cold_start_new_format_av-bs36-ga3-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-200
# Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_10-22-0705-emer-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-55

# EMER-MERR 0.5B : /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0525-EMER-MERR-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower
#7B:  /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0605-EMER-MERR-8frame-all_in_one-HumanOmni-EPOCH10-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-750


MODEL_ABBR=TRI-av-SFT-0.5B
REWARD_LIST=("av_format" "accuracy" "think_av_with_neutral")
REWARD_ABBR=$(
  for item in "${REWARD_LIST[@]}"; do
    # 언더스코어로 나누고 첫 글자씩 추출
    IFS='_' read -ra parts <<< "$item"
    abbr=""
    for p in "${parts[@]}"; do
      abbr+="${p:0:1}"
    done
    printf "%s_" "$abbr"
  done | sed 's/_$//'
)

echo "$REWARD_ABBR"

# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0601-EMER-MERR-fr32-HumanOmni-EPOCH5-LR1e-4-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-745

SAVE_STRATEGY="steps"
SAVE_STEPS=100



RUN_CONCEPT=${CURRENT_TIME}-${MODEL_ABBR}-${REWARD_ABBR}
RUN_NAME=results-${RUN_CONCEPT}-epoch${train_epoch}-ratio${data_ratio}-grpo${num_generations}-lr${LR}-bs${batch_size}-gradaccum${grad_accum}-beta${beta}-layer${select_layer_idx}
OUTPUT_DIR=${SRC_DIR}/Exp/R1-Omni/src/r1-v/${RUN_NAME}
MODEL_DIR=${MODEL_NAME}
DATA_SRC=${SRC_DIR}/Dataset
DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-AV.json
TEST_DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-FINE.json


WANDB_MODE=online torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12114" \
    /mnt/ssd_hs/Exp/R1-Omni/src/r1-v/src/open_r1/grpo-text_av_attention_AU.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_DIR} \
    --dataset_name ${DATASET_NAME} \
    --test_dataset_name ${TEST_DATASET_NAME} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 256 \
    --max_completion_length 350 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum} \
    --per_device_eval_batch_size 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs ${train_epoch} \
    --run_name ${RUN_NAME} \
    --save_strategy=${SAVE_STRATEGY} \
    --save_steps=${SAVE_STEPS} \
    --save_only_model false \
    --num_generations ${num_generations} \
    --reward_funcs "${REWARD_LIST[@]}" \
    --reference_model_switch true \
    --data_ratio ${data_ratio} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --learning_rate ${LR} \
    --dataloader_num_workers 8 \
    --beta ${beta} \
    --select_layer_idx ${select_layer_idx} \


