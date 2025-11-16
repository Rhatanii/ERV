#!/bin/bash
conf=/mnt/ssd_hs/Exp/R1-Omni/configs/inference-7B.yaml
# /mnt/ssd_hs/Exp/R1-Omni/configs/inference_text_only.yaml

CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/mnt/ssd_hs/.cache python inference.py --config ${conf}