#!/bin/bash
MODEL_BASE_NAME=HumanOmni #R1-Omni
MODEL_NAME=StarJiaxing/${MODEL_BASE_NAME}-0.5B


accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill