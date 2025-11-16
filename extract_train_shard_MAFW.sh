#!/bin/bash
DATASET_NAME=MAFW

conf=/mnt/ssd_hs/Exp/R1-Omni/configs/train/train_${DATASET_NAME}-7B.yaml
model_name=Human-Omni-7B
shard=7
output_file=/mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_train5_all.txt
modal=video_audio

for i in $(seq 0 $shard)
do
    CUDA_ID=$((i % 8))
    echo "Running shard $i"
    export CUDA_VISIBLE_DEVICES=$CUDA_ID
    export HF_HUB_CACHE=/mnt/ssd_hs/.cache 
    python eval_shard.py --config ${conf} --modal $modal --shards $((shard + 1)) --shard_id ${i} --output_path  /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_train5_${i}.txt &
done

wait
for i in $(seq 0 $shard); do
    cat /mnt/ssd_hs/Exp/R1-Omni/results/${DATASET_NAME}/${model_name}/output_train5_${i}.txt >> $output_file
done
