#!/bin/bash

port=$(python tools/get_free_port.py)
GPU=2
CUDA_VISIBLE_DEVICES=0,1

# Joint-Training for all data
python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/voc/e2e_faster_rcnn_R_50_C4_4x_JT.yaml -cvd ${CUDA_VISIBLE_DEVICES}