# !/bin/bash

data_type="test"
# Which dataset to process ("DFEW" "MAFW")
dataset_name_list=("MAFW" "DFEW") 

nshard=20 # number of shards
echo "$dataset_name"
model_id_list=("ERV-7B_neutral/checkpoint-1044" )


for dataset_name in "${dataset_name_list[@]}"
do
    instruction_label_name=$dataset_name
    for model_id in "${model_id_list[@]}"
    do
        echo "Processing dataset: $dataset_name with model: $model_id"

        for i in $(seq 0 $((nshard-1)))
        do
            python response_gpt_check.py \
                --dataset_name ${dataset_name} \
                --data_type ${data_type} \
                --nshard ${nshard} \
                --shard_id ${i} \
                --model_id ${model_id} \
                --instruction_label_name ${instruction_label_name} &
            sleep 2
        done
        wait

        output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${dataset_name}/${model_id}/label_false/output_eval5_all-video_audio-instruct_${instruction_label_name}


        echo "Concat all shards"
        python response_gpt_check-merge.py --dataset_name ${dataset_name} \
            --data_type ${data_type} \
            --nshard ${nshard} \
            --model_id ${model_id} \
            --instruction_label_name ${instruction_label_name}

        wait
        echo "Merged JSON saved "
        for i in $(seq 0 $((nshard-1)))
        do
            rm -rf ${output_file}-${i}.json
        done
    done
done
