#!/bin/bash
DATASET_NAME=MAFW-7B
conf=/mnt/ssd_hs/Exp/R1-Omni/configs/eval/eval_${DATASET_NAME}.yaml

HF_HUB_CACHE=/mnt/ssd_hs/.cache python eval.py --config ${conf}