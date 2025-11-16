cd src/r1-v

export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/mnt/ssd_hs/.cache
export HF_HUB_CACHE=/mnt/ssd_hs/.cache
mkdir -p ./logs

echo "Log path set to: $LOG_PATH"


OUTPUT_DIR=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v/results-0416-sft-rlvr
MODEL_DIR=/mnt/ssd_hs/Exp/R1-Omni/work_dirs/humanomniqwen2_siglip/finetune_EMER_SFT
DATA_SRC=/mnt/ssd_hs/Dataset
DATASET_NAME=/mnt/ssd_hs/Dataset/R1-Omni/RLVR.json


WANDB_MODE=offline torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_DIR} \
    --dataset_name ${DATASET_NAME} \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-emotion \
    --save_steps 500 \
    --save_only_model true \
    --num_generations 4   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  