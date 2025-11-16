#!/bin/bash
data_type=eval
total_shard=27
# use_SFT=false
# /mnt/ssd_hs/Exp/R1-Omni/work_dirs/brev/AU-SFT_a_f_ts_ekaat_bs1_ga2_lr1e-6_epoch2/checkpoint-450
MODEL_PATH_ABBR_LIST=("humanomniqwen2_siglip/finetune_11-06-2025-EMER_MAFW_DFEW-ERV_cold_start_new_format_av-bs36-ga3-HumanOmni-EPOCH5-LR2e-5-mm_language_model,mm_mlp_adapter,audio_projector,mm_vision_tower/checkpoint-200" )

DATA_LIST=("MAFW" "DFEW")
LABEL_LIST=("false")
modal="video_audio"
use_AU=false
use_AU01=false
use_AU01_w_table=false
use_AU01_w_info=false
use_special_format=false
use_new_av_format=True

do_sample=true
temperature_list=(0.3)


for temperature in "${temperature_list[@]}"
do
    for MODEL_PATH_ABBR in "${MODEL_PATH_ABBR_LIST[@]}"
    do
        MODEL_SRC=/mnt/ssd_hs/Exp/R1-Omni/work_dirs
        MODEL_PATH=${MODEL_SRC}/${MODEL_PATH_ABBR}
        for DATASET_NAME in "${DATA_LIST[@]}"
        do
            for with_label in "${LABEL_LIST[@]}"
            do
                if [ "$with_label" = false ]; then
                    instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                    # "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                elif [ "$with_label" = true ] && [ "$DATASET_NAME" = DFEW ];then
                    instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?\n happy ,surprise ,neutral ,angry ,disgust ,sad ,fear. Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                elif [ "$with_label" = true ] && [ "$DATASET_NAME" = MAFW ];then
                    instruction="As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?\nhappiness ,surprise ,neutral , anger ,disgust ,sadness ,fear ,contempt, disappointment, helplessness, anxiety.  Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
                fi


                conf=/mnt/ssd_hs/Exp/R1-Omni/configs/${data_type}/${data_type}_${DATASET_NAME}.yaml
                model_name=${MODEL_PATH_ABBR}/label_${with_label}
                shard=${total_shard}
                output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_all-${modal}-tmp${temperature}.txt

                for i in $(seq 0 $shard)
                do
                    CUDA_ID=$((i % 4))
                    echo "Running shard $i"
                    if [ "$CUDA_ID" -ge 0 ]; then
                        CUDA_ID=$((CUDA_ID + 4))
                    fi
                    export CUDA_VISIBLE_DEVICES=$CUDA_ID
                    export HF_HUB_CACHE=/mnt/ssd_hs/.cache 
                    python eval_shard.py --config ${conf} --modal ${modal} --shards $((shard + 1)) --shard_id ${i} --instruct "${instruction}" --model_path ${MODEL_PATH} --output_path /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}-tmp${temperature}.txt --use_AU ${use_AU} --use_AU01 ${use_AU01}  --use_AU01_w_table ${use_AU01_w_table} --use_AU01_w_info ${use_AU01_w_info} --use_special_format ${use_special_format} --use_new_av_format ${use_new_av_format} --do_sample ${do_sample} --temperature ${temperature} & 
                    sleep 1
                    
                done

                wait
                for i in $(seq 0 $shard); do
                    cat /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}-tmp${temperature}.txt >> $output_file
                done

                wait
                for i in $(seq 0 $shard); do
                    rm /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}-tmp${temperature}.txt
                done


            done
            wait
        done
        wait
    done
done
