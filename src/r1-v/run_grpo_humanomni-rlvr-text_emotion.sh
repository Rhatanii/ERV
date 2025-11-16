cd src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/mnt/ssd_hs/.cache
export HF_HUB_CACHE=/mnt/ssd_hs/.cache
mkdir -p ./logs

echo "Log path set to: $LOG_PATH"

train_epoch=1
data_ratio=1
eval_strategy=no # no steps epoch
eval_steps=1
num_generations=4
LR=1e-6
BETA=0.1
reference_model_switch=false

LR_SCHEDULER=cosine
MODEL_NAME=/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0515-EMER-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower

# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0525-EMER-MERR-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower


# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0605-EMER-without_reason-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-22

# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0525-EMER-MERR-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-150

# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0601-EMER-MERR-fr32-HumanOmni-EPOCH5-LR1e-4-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-745
# EMER
# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0515-EMER-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-55

# /mnt/ssd_hs/Exp/R1-Omni/src/r1-v/results-0522-acc-format-think_confusion_with_hint-r1-omni-RLVR-MAFW-single-epoch4-ratio1-grpo4-lr1e-6-constant/checkpoint-1060
# StarJiaxing/R1-Omni-0.5B

# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0515-EMER-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-55
# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0515-EMER-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-55
# StarJiaxing/R1-Omni-0.5B
REWARD_LIST=("format" "accuracy" "think")

lora_enabled=false
use_peft=false
rank=64
alpha=64
CURRENT_TIME=$(date +"%m-%d-%H%M")
MODEL_ABBR=EMER-SFT-REF_${reference_model_switch}
REWARD_ABBR=FORMAT_ACC_THINK


RUN_CONCEPT=${CURRENT_TIME}-${MODEL_ABBR}-${REWARD_ABBR}-RLVR-fr8-beta${BETA}-lora_${lora_enabled}-r${rank}-a${alpha}



RUN_NAME=results-${RUN_CONCEPT}-epoch${train_epoch}-ratio${data_ratio}-grpo${num_generations}-lr${LR}-${LR_SCHEDULER}
OUTPUT_DIR=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v/${RUN_NAME}
MODEL_DIR=${MODEL_NAME}
DATA_SRC=/mnt/ssd_hs/Dataset
DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR.json
TEST_DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-INSTRUCT-MAFW-TEST.json


WANDB_MODE=online torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo-text.py \
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
    --data_ratio ${data_ratio} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \
    --learning_rate ${LR} \
    --lr_scheduler_type ${LR_SCHEDULER} \
    --warmup_ratio 0.05 \
    --beta ${BETA} \
    --resume_from_checkpoint ${MODEL_DIR} \
    --lora_enabled ${lora_enabled} \
    --lora_r $rank \
    --lora_alpha $alpha \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --use_peft ${use_peft} \
    --reference_model_switch ${reference_model_switch}

        # --save_steps 500 \