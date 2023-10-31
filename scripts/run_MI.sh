#!/bin/bash

port=$(python tools/get_free_port.py)
GPU=2

alias exp="python -m torch.distributed.launch --master_port=${port} --nproc_per_node=${GPU} tools/train_incremental.py"
shopt -s expand_aliases


# INCREMENTAL STEPS
#task=10-5
#name=ABR_LR001_BS4_ALPHA1_BETA1_GAMMA1
#mb=2000
#mt=mean
#
#for s in {1..2};
#do
#    exp -t ${task} -n ${name} -s $s --feat afd -gamma 1.0 --uce --dist_type id -alpha 1.0 -beta 1.0 -mb $mb -mt $mt -cvd 0,1
#    python tools/prototype_box_selection.py -cvd 0 -n ${name} -t ${task} -s $s -mb $mb -mt $mt -iss
#  echo Done
#done


#task=5-5
#name=ABR_LR001_BS4_ALPHA05_BETA1_GAMMA1
#for s in {1..4};
#do
#    exp -t ${task} -n ${name} -s $s --feat afd -gamma 1.0 --uce --dist_type id -alpha 0.5 -beta 1.0 -mb 2000 -mt mean -cvd 0,1
#    python tools/prototype_box_selection.py -cvd 0 -n ${name} -t ${task} -s $s -mb 2000 -mt mean -iss
#  echo Done
#done


#task=10-2
#name=ABR_LR002_BS4_ALPHA1_BETA1_GAMMA05
#mb=2000
#mt=mean
#
#for s in {1..5};
#do
#    exp -t ${task} -n ${name} -s $s --feat afd -gamma 0.5 --uce --dist_type id -alpha 1.0 -beta 1.0 -mb $mb -mt $mt -cvd 0,1
#    python tools/prototype_box_selection.py -cvd 0 -n ${name} -t ${task} -s $s -mb $mb -mt $mt -iss
#  echo Done
#done


#task=15-1
#name=ABR_LR001_BS4_ALPHA1_BETA1_GAMMA5
#mb=2000
#mt=mean
#
#for s in {1..5};
#do
#    exp -t ${task} -n ${name} -s $s --feat afd -gamma 1.0 --uce --dist_type id -alpha 1.0 -beta 1.0 -mb $mb -mt $mt -cvd 0,1
#    python tools/prototype_box_selection.py -cvd 0 -n ${name} -t ${task} -s $s -mb $mb -mt $mt -iss
#  echo Done
#done


task=10-1
name=ABR_LR002_BS4_ALPHA1_BETA1_GAMMA1
mb=2000
mt=mean

for s in {1..10};
do
    exp -t ${task} -n ${name} -s $s --feat afd -gamma 1.0 --uce --dist_type id -alpha 1.0 -beta 1.0 -mb $mb -mt $mt -cvd 0,1
    python tools/prototype_box_selection.py -cvd 0 -n ${name} -t ${task} -s $s -mb $mb -mt $mt -iss
  echo Done
done