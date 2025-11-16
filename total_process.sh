#!/bin/bash


for EPOCH in 1
do
    for LR in 2e-5 1e-5
    do
        bash scripts/finetune_omni_add_reasoning.sh 1 8 ${EPOCH} $LR

        bash eval_shard-SFT-CLUE.sh ${EPOCH} $LR

        python eval_score-clues.py --epoch ${EPOCH} --lr ${LR}
    done
done