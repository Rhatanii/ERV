#!/bin/bash
data_type=eval
total_shard=7

MODEL_PATH_ABBR_LIST=("/mnt/ssd_hs/Exp/R1-Omni/pre-trained/HumanOmni-7B") #"work_dirs/humanomniqwen2_siglip/checkpoint-150")
DATA_LIST=("EMER" )
do_sample=True
temperature=0.3
# /mnt/ssd_hs/Exp/R1-Omni/src/r1-v/results-0513-text-emotion-epoch1-ratio0.2-grpo4-lr1e-6
for MODEL_PATH_ABBR in "${MODEL_PATH_ABBR_LIST[@]}"
do
    MODEL_SRC=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v
    # MODEL_PATH=${MODEL_SRC}/${MODEL_PATH_ABBR}
    MODEL_PATH=${MODEL_PATH_ABBR}
    for DATASET_NAME in "${DATA_LIST[@]}"
    do
        for with_label in true
        do
            if [ "$with_label" = false ]; then
                instruction="As an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\n"
            elif [ "$with_label" = true ] && [ "$DATASET_NAME" = DFEW ];then
                instruction="As an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\nhappy ,surprise ,neutral ,angry ,disgust ,sad ,fear"
            elif [ "$with_label" = true ] && [ "$DATASET_NAME" = MAFW ];then
                instruction="As an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\nhappy ,surprise ,neutral ,angry ,disgust ,sad ,fear ,contemptuous, disappointed, helpless, anxious"
            elif [ "$with_label" = true ] && [ "$DATASET_NAME" = EMER ];then
                instruction="As an emotional recognition expert, in the video, when the characters display their emotions, which predominant feeling is most clearly expressed?\nhappy ,surprise ,angry ,sad , worried"
            fi
            # instruction="What is the emotional state of the person in the video? Please tell me the reason."
            # for modal in video_audio
            for modal in video_audio audio video
            do
                conf=/mnt/ssd_hs/Exp/R1-Omni/configs/${data_type}/${data_type}_${DATASET_NAME}-7B.yaml
                model_name=${MODEL_PATH_ABBR}/label_${with_label}
                shard=${total_shard}
                output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_all-${modal}.txt

                for i in $(seq 0 $shard)
                do
                    CUDA_ID=$((i % 8))
                    echo "Running shard $i"
                    export CUDA_VISIBLE_DEVICES=$CUDA_ID
                    export HF_HUB_CACHE=/mnt/ssd_hs/.cache 
                    python eval_shard.py --config ${conf} --modal ${modal} --shards $((shard + 1)) --shard_id ${i} --instruct "${instruction}" --model_path ${MODEL_PATH} --do_sample ${do_sample} --temperature ${temperature} --output_path /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt &
                    sleep 1
                done
                wait
                for i in $(seq 0 $shard)
                do
                    cat /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt >> $output_file
                done
                wait

                for i in $(seq 0 $shard); do
                    rm /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt
                done
                wait
            done
            wait

        done
        wait
    done
    wait
done

# wait

# for MODEL_PATH_ABBR in "${MODEL_PATH_ABBR_LIST[@]}"
# do
#     MODEL_SRC=/mnt/ssd_hs/Exp/R1-Omni/src/r1-v
#     MODEL_PATH=${MODEL_SRC}/${MODEL_PATH_ABBR}
#     for DATASET_NAME in "${DATA_LIST[@]}"
#     do
#         for with_label in false
#         do
#             for modal in video_audio
#             do
#                 shard=${total_shard}
#                 model_name=${MODEL_PATH_ABBR}/label_${with_label}
#                 output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_all-${modal}.txt

#                 for i in $(seq 0 $shard); do
#                     cat /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt >> $output_file
#                 done


#                 for i in $(seq 0 $shard); do
#                     rm /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_${data_type}5_${i}-${modal}.txt
#                 done
#             done
#         done
#     done
# done