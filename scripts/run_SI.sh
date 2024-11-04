#!/bin/bash

port=$(python tools/get_free_port.py)
GPU=2

alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py"
shopt -s expand_aliases


# INCREMENTAL STEPS
step=1

#task=19-1
#name=ABR_LR001_BS4_ALPHA1_BETA1_GAMMA5
#exp -t ${task} -n ${name} -s ${step} --feat ard -gamma 5.0 --uce --dist_type id -alpha 1.0 -beta 1.0 -mb 2000 -mt mean -cvd 0,1
#name=Finetune
#exp -t ${task} -n ${name} -s ${step} -cvd 0,1


task=15-5
name=ABR_LR001_BS4_ALPHA05_BETA1_GAMMA1
exp -t ${task} -n ${name} -s ${step} --feat ard -gamma 1.0 --uce --dist_type id -alpha 0.5 -beta 1.0 -mb 2000 -mt mean -cvd 0,1
#name=Finetune
#exp -t ${task} -n ${name} -s ${step} -cvd 0,1


#task=10-10
#name=ABR_LR005_BS4_ALPHA01_BETA05_GAMMA1
#exp -t ${task} -n ${name} -s ${step} --feat ard -gamma 1.0 --uce --dist_type id -alpha 0.1 -beta 0.5 -mb 2000 -mt mean -cvd 0,1
#name=Finetune
#exp -t ${task} -n ${name} -s ${step} -cvd 0,1


#task=5-15
#name=ABR_LR005_BS4_ALPHA01_BETA05_GAMMA1
#exp -t ${task} -n ${name} -s ${step} --feat ard -gamma 1.0 --uce --dist_type id -alpha 0.1 -beta 0.5 -mb 2000 -mt mean -cvd 0,1
#name=Finetune
#exp -t ${task} -n ${name} -s ${step} -cvd 0,1


