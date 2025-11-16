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
num_generations=2
LR=1e-6
LR_SCHEDULER=cosine


MODEL_NAME=EMER-SFT # EMER-SFT, EMER-MERR-SFT, R1-Omni-0.5B
MODEL_ABBR=${MODEL_NAME}
REWARD_LIST=("format" "accuracy" "think_step")
REWARD_ABBR=FORMAT_ACC_THINK-STEP
DATASET_ABBR=RLVR # RLVR, RLVR-FINE

lora_enabled=false
use_peft=false
rank=64
alpha=64
CURRENT_TIME=$(date +"%m-%d-%H%M")


if [ "$MODEL_NAME" = "EMER-SFT" ]; then
  MODEL_PATH="/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0515-EMER-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower"
elif [ "$MODEL_NAME" = "EMER-MERR-SFT" ]; then
  MODEL_PATH="/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_0525-EMER-MERR-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower"
elif [ "$MODEL_NAME" = "MERR-SFT" ]; then
  MODEL_PATH="/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_07-18-0446-MERR-FINE-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower"
elif [ "$MODEL_NAME" = "R1-Omni-0.5B" ]; then
  MODEL_PATH="StarJiaxing/R1-Omni-0.5B"
else
  echo "Error: Can't process MODEL_NAME: $MODEL_NAME"
  exit 1
fi



RUN_CONCEPT=${CURRENT_TIME}-${MODEL_ABBR}-${REWARD_ABBR}-${DATASET_ABBR}

RUN_NAME=results-${RUN_CONCEPT}-epoch${train_epoch}-ratio${data_ratio}-grpo${num_generations}-lr${LR}-${LR_SCHEDULER}
OUTPUT_DIR=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v/${RUN_NAME}
DATA_SRC=/mnt/ssd_hs/Dataset
DATASET_NAME=${DATA_SRC}/R1-Omni/${DATASET_ABBR}.json
TEST_DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-INSTRUCT-MAFW-TEST.json


WANDB_MODE=online torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo-text_multi_verifier.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATASET_NAME} \
    --test_dataset_name ${TEST_DATASET_NAME} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
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
    --resume_from_checkpoint ${MODEL_PATH} \
    --lora_enabled ${lora_enabled} \
    --lora_r $rank \
    --lora_alpha $alpha \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --use_peft ${use_peft} \

        # --save_steps 500 \