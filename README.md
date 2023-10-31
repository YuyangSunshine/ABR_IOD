# Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection -- @ICCV 23, [Paper](https://ui.adsabs.harvard.edu/link_gateway/2023arXiv230712427Y/arxiv:2307.12427)

# Abstract
![abs](https://github.com/YuyangSunshine/ABR_IOD/assets/40997704/105473af-b76f-4d88-9a77-872aa8f88e09)

In incremental learning, replaying stored samples from previous tasks together with current task samples is one of the most efficient approaches to address catastrophic forgetting. However, unlike incremental classification, image replay has not been successfully applied to incremental object detection (IOD). In this paper, we identify the overlooked problem of foreground shift as the main reason for this. Foreground shift only occurs when replaying images of previous tasks and refers to the fact that their background might contain foreground objects of the current task. To overcome this problem, a novel and efficient Augmented Box Replay (ABR) method is developed that only stores and replays foreground objects and thereby circumvents the foreground shift problem. In addition, we propose an innovative Attentive RoI Distillation loss that uses spatial attention from region-of-interest (RoI) features to constrain current model to focus on the most important information from old model. ABR significantly reduces forgetting of previous classes while maintaining high plasticity in current classes. Moreover, it considerably reduces the storage requirements when compared to standard image replay. Comprehensive experiments on Pascal-VOC and COCO datasets support the state-of-the-art performance of our model.

# Overview
![method](https://github.com/YuyangSunshine/ABR_IOD/assets/40997704/0ad80f4f-5920-43e1-ba1e-e3e1b19a616a)
Illustration of our proposed framework, which highlights the key novelties of Augmented Box Replay (ABR) and Attentive RoI Distillation. ABR fuses prototype object $b$ from Box Rehearsal $B^{t-1}$ into the current image $I_n^t$ using mixup or mosaic. Attentive RoI Distillation uses pooled attention $A_i$ and masked features $F_i \cdot A_i^{t-1}$ to constrain the model to focus on important information from previous model. Inclusive Distillation Loss overcomes catastrophic forgetting based on ABR.

# How to run
## Install
Please, follow the instruction provided by Detectron 1 and found in install.md

## Dataset
You can find the Pascal-VOC dataset already in Detectron.

## Setting!
We provide scripts to run the experiments in the paper (JT, FT, ABR and ablations).
You can find three scripts in the `scripts/` file: `run_JT.sh`,  `run_MI.sh`, and `run_SI.sh`. The file can be used to run, respectively: single-incremetal-step detection settings (19-1, 15-5, 10-10, 5-15), multi-incremental-step detection settings (10-5, 10-2, 15-1, 10-1, 5-5).

Without specifying any option, the defaults will load the Finetune method using the Faster-RCNN. 
You can play with the following parameters to obtain all the results in the paper:
- `--feat` with options [`no`, `std`, `ard`]. No means not using feature distillation, `std` is the feature distillation employed in Faster-ILOD, while `ard` is the attentive RoI distillation (as in ABR)- (default: no).
- `--inc` will enable the incremental setting - (default: not use);
- `--dist_type` with options [`l2`, `id`, `none`], where `l2` is the distillation used in ILOD, `id` the Inclusive Distillation Loss used in our method ABR, and `none` means not use it (default: l2);
- `--alpha_inclusive_distillation` is a float indicating the weight of the inclusive_distillation loss (default: 1.); In ABR we vary it in range [0.1, 0.2, 0.5, 1];
- `--beta_attentive_roi_distillation` is a float indicating the weight of the attentive RoI distillation loss.
- `--gamma` is a hyperparameter that controls the strength of the regularization of overall ARD loss. We used 1.0 for our ABR.
- `--memory_buffer` is a int representing the box rehearsal memory size.
- `--memory_type` with options [`mean`, `random`, `herding`], where `mean` is the replay strategy in our ABR, `random` is choosing the boxes images randomly, while `herding` selecting boxes by herding strategy in icarl - (default: None). 

## Run!
#### 1. Train the first task

For the setting of the Pascal-VOC dataset, the first task usually contains categories 5, 10, 15 and 19. 
So when we train the first task for different settings, we could only train four kinds of the first task.
First, choosing the correspoinding task setting (e.g., `taks=10-10`) in the `scripts/run_firststep.sh` file according to your needs, and then run:


#### 2. Select the box rehearsal for the first task

To avoid repeated calculations with the same first task, we place the box rehearsal in the output root of the first task, and then run:

``` shell script
bash scripts/run_firststep.sh
``` 

#### 3. Train the single/multiple incremental setting task

Modify the `scripts/run_SI.sh` or `scripts/run_MI.sh`. 
For example, if you want to train ABR under single incremental step `task=15-5`, then please keep the following sentences uncommented:
``` shell script
task=15-5
name=ABR_LR001_BS4_ALPHA05_BETA1_GAMMA1_INIT
exp -t ${task} -n ${name} -s ${step} --feat afd -gamma 1.0 --uce --dist_type id -alpha 0.5 -beta 1.0 -mb 2000 -mt mean -cvd 0,1
``` 
, and then run:

``` shell script
bash scripts/run_SI.sh
``` 

Otherwise, if you want to train the finetune method, then please keep the following sentences uncommented:

``` shell script
task=15-5
name=Finetune
exp -t ${task} -n ${name} -s ${step} -cvd 0,1
``` 
, and then run:

``` shell script
bash scripts/run_SI.sh
``` 

# Cite us!
``` 
@InProceedings{yuyang2023augmented,
  title={Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection},
  author={Yuyang, Liu and Yang, Cong and Dipam, Goswami and Xialei, Liu and van de Weijer, Joost},
  booktitle={In Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  month={October},
  year={2023}
}
```
# Acknowledgments
Our repository is based on the amazing work of @fcdl94[MMA](https://github.com/fcdl94/MMA) and @CanPeng123 [FasterILOD](https://github.com/CanPeng123/Faster-ILOD) and on the [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) library. We thank the authors and the contibutors of these projects for releasing their code.
