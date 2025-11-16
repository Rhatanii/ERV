cd src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/mnt/ssd_hs/.cache
export HF_HUB_CACHE=/mnt/ssd_hs/.cache
mkdir -p ./logs

echo "Log path set to: $LOG_PATH"

train_epoch=1
data_ratio=0.1
eval_strategy=no # no steps epoch
eval_steps=1
num_generations=4


RUN_CONCEPT=MAFW-w-neutral-start_from_no_reason
RUN_NAME=results-0508-${RUN_CONCEPT}-epoch${train_epoch}-ratio${data_ratio}-grpo${num_generations}
OUTPUT_DIR=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v/${RUN_NAME}
MODEL_DIR=/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_MAFW-w-neutral-HumanOmni-EPOCH1-LR2e-5-audio_projector,mm_mlp_adapter,mm_language_model
DATA_SRC=/mnt/ssd_hs/Dataset
DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-FINE.json
TEST_DATASET_NAME=${DATA_SRC}/R1-Omni/RLVR-INSTRUCT-MAFW-TEST.json


WANDB_MODE=offline torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
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
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs ${train_epoch} \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --save_only_model true \
    --num_generations ${num_generations} \
    --reference_model_switch true \
    --data_ratio ${data_ratio} \
    --eval_strategy ${eval_strategy} \
    --eval_steps ${eval_steps} \

