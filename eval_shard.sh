#!/bin/bash

SRC_DIR=/mnt/ssd_hs/Exp/R1-Omni
data_type=eval
total_shard=7

MODEL_PATH_ABBR_LIST=("ERV-7B/checkpoint-1044")
use_7B_model=true # true or false

DATA_LIST=( "DFEW" "MAFW")
temperature_LIST=(0.3)
do_sample=True

for temperature in "${temperature_LIST[@]}"
do
    echo "Temperature: $temperature"
    for MODEL_PATH_ABBR in "${MODEL_PATH_ABBR_LIST[@]}"
    do
        MODEL_SRC=${SRC_DIR}/src/r1-v/checkpoints
        MODEL_PATH=${MODEL_SRC}/${MODEL_PATH_ABBR}
        for DATASET_NAME in "${DATA_LIST[@]}"
        do
            instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."

            if [ "$use_7B_model" = true ]; then
                conf=/mnt/ssd_hs/Exp/R1-Omni/configs/${data_type}/${data_type}_${DATASET_NAME}-7B.yaml
            else
                conf=/mnt/ssd_hs/Exp/R1-Omni/configs/${data_type}/${data_type}_${DATASET_NAME}.yaml
            fi

            model_name=${MODEL_PATH_ABBR}/label_false
            shard=${total_shard}
            output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_all-video_audio-tmp${temperature}.txt

            for i in $(seq 0 $shard)
            do
                CUDA_ID=$((i % 8))
                echo "Running shard $i"
                export CUDA_VISIBLE_DEVICES=$CUDA_ID
                export HF_HUB_CACHE=/mnt/ssd_hs/.cache 
                python eval_shard.py --config ${conf} --modal video_audio --shards $((shard + 1)) --shard_id ${i} --instruct "${instruction}" --model_path ${MODEL_PATH} --output_path ${SRC_DIR}/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-video_audio.txt --temperature ${temperature} --do_sample ${do_sample} &
                sleep 1
            done
            wait

            # Merge shard outputs
            for i in $(seq 0 $shard)
            do
                cat ${SRC_DIR}/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-video_audio.txt >> $output_file
            done

            # Clean up shard files
            for i in $(seq 0 $shard); do
                rm ${SRC_DIR}/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-video_audio.txt
            done
            wait
        done
        wait
    done
    wait
done
