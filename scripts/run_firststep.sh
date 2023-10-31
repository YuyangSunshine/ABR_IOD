#!/bin/bash
port=$(python tools/get_free_port.py)
GPU=2
CUDA_VISIBLE_DEVICES=0,1

#  Choose the corresponding first task #
# 19 classes in first task #
#task=19-1

# 15 classes in first task #
task=15-5

# 10 classes in first task #
#task=10-10

# 5 classes in first task #
# task=5-15


#### 1. Train the First Task ####

python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_first_step.py -c configs/voc/${task}/e2e_faster_rcnn_R_50_C4_4x.yaml -cvd ${CUDA_VISIBLE_DEVICES}

#### 2. Prototype Box Selection (PBR) ####
step=0
name=ABR # The corrensponding name for the incremental method
python tools/prototype_box_selection.py \
    -cvd 0 \
    -n ${name} -t ${task} -s ${step} \
    -mb 2000 \
    -mt mean \
    -iss