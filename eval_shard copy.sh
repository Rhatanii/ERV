#!/bin/bash
data_type=eval
total_shard=7

MODEL_PATH_ABBR_LIST=("ERV-7B_neutral/checkpoint-1044")

use_7B_model=true # true or false

DATA_LIST=( "DFEW" "MAFW")
LABEL_LIST=("false" )
use_only_AU_matched_dataset=False
temperature_LIST=(0.3)
do_sample=True

instruction="As an emotion recognition expert, analyze the video considering visual behavior, and audio tone. Describe your observations using the tags <vis_desc> and <aud_desc> for visual, and audio cues respectively. Focus first on the modality that provides the most important information for emotion prediction. Then, integrate all these cues into a reasoning process in <think> </think>, and provide the final predicted emotion in <answer> </answer> tags."

for temperature in "${temperature_LIST[@]}"
do
    echo "Temperature: $temperature"
    for MODEL_PATH_ABBR in "${MODEL_PATH_ABBR_LIST[@]}"
    do
        MODEL_SRC=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v/checkpoints
        MODEL_PATH=${MODEL_SRC}/${MODEL_PATH_ABBR}
        # MODEL_PATH=${MODEL_PATH_ABBR}/
        for DATASET_NAME in "${DATA_LIST[@]}"
        do
            for with_label in false
            do
                # --- 조건문 추가 부분 ---
                # if [[ "$temperature" == "1" && "$with_label" == "true" && "$DATASET_NAME" == "MAFW" ]]; then
                #     echo "Skipping case: temperature=$temperature, with_label=$with_label, DATASET_NAME=$DATASET_NAME"
                #     continue
                # fi

                if [ "$with_label" = false ]; then
                    instruction="As an emotion recognition expert, analyze the video considering visual behavior, and audio tone. Describe your observations using the tags <vis_desc> and <aud_desc> for visual, and audio cues respectively. Focus first on the modality that provides the most important information for emotion prediction. Then, integrate all these cues into a reasoning process in <think> </think>, and provide the final predicted emotion in <answer> </answer> tags."
                    # instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                elif [ "$with_label" = true ] && [ "$DATASET_NAME" = DFEW ];then
                    instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?\n happy ,surprise ,neutral ,angry ,disgust ,sad ,fear. Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                elif [ "$with_label" = true ] && [ "$DATASET_NAME" = MAFW ];then
                    instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?\nhappiness ,surprise ,neutral , anger ,disgust ,sadness ,fear ,contempt, disappointment, helplessness, anxiety.  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                fi

                for modal in video_audio
                do
                    if [ "$use_7B_model" = true ]; then
                        conf=/mnt/ssd_hs/Exp/R1-Omni/configs/${data_type}/${data_type}_${DATASET_NAME}-7B.yaml
                    else
                        conf=/mnt/ssd_hs/Exp/R1-Omni/configs/${data_type}/${data_type}_${DATASET_NAME}.yaml
                    fi

                    model_name=${MODEL_PATH_ABBR}/label_${with_label}
                    shard=${total_shard}
                    output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_all-${modal}-tmp${temperature}.txt

                    for i in $(seq 0 $shard)
                    do
                        CUDA_ID=$((i % 8))
                        echo "Running shard $i"
                        export CUDA_VISIBLE_DEVICES=$CUDA_ID
                        export HF_HUB_CACHE=/mnt/ssd_hs/.cache 
                        python eval_shard.py --config ${conf} --modal ${modal} --shards $((shard + 1)) --shard_id ${i} --instruct "${instruction}" --model_path ${MODEL_PATH} --output_path /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt --temperature ${temperature} --do_sample ${do_sample} --use_only_AU_matched_dataset ${use_only_AU_matched_dataset} &
                        sleep 1
                    done
                done
                wait
                for i in $(seq 0 $shard)
                do
                    cat /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt >> $output_file
                done


                for i in $(seq 0 $shard); do
                    rm /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt
                done
            done
            wait
        done
        wait
    done
    wait
done
