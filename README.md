# Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection -- @ICCV 23, [Paper](https://ui.adsabs.harvard.edu/link_gateway/2023arXiv230712427Y/arxiv:2307.12427)

# Abstract
![problem](https://github.com/YuyangSunshine/ABR_IOD/assets/40997704/bfd6cbca-e8f7-4485-9ca0-98155c3d7c54)
![10-1](https://github.com/YuyangSunshine/ABR_IOD/assets/40997704/b7c71223-6c1b-498e-95d7-730b6fa49c23)
![5-5](https://github.com/YuyangSunshine/ABR_IOD/assets/40997704/08d1d526-50e4-4a52-bb46-deb15ca5ce57)

In incremental learning, replaying stored samples from previous tasks together with current task samples is one of the most efficient approaches to address catastrophic forgetting. However, unlike incremental classification, image replay has not been successfully applied to incremental object detection (IOD). In this paper, we identify the overlooked problem of foreground shift as the main reason for this. Foreground shift only occurs when replaying images of previous tasks and refers to the fact that their background might contain foreground objects of the current task. To overcome this problem, a novel and efficient Augmented Box Replay (ABR) method is developed that only stores and replays foreground objects and thereby circumvents the foreground shift problem. In addition, we propose an innovative Attentive RoI Distillation loss that uses spatial attention from region-of-interest (RoI) features to constrain current model to focus on the most important information from old model. ABR significantly reduces forgetting of previous classes while maintaining high plasticity in current classes. Moreover, it considerably reduces the storage requirements when compared to standard image replay. Comprehensive experiments on Pascal-VOC and COCO datasets support the state-of-the-art performance of our model.

# Overview
![method](https://github.com/YuyangSunshine/ABR_IOD/assets/40997704/0ad80f4f-5920-43e1-ba1e-e3e1b19a616a)
Illustration of our proposed framework, which highlights the key novelties of Augmented Box Replay (ABR) and Attentive RoI Distillation. ABR fuses prototype object $b$ from Box Rehearsal $B^{t-1}$ into the current image $I_n^t$ using mixup or mosaic. Attentive RoI Distillation uses pooled attention $A_i$ and masked features $F_i \cdot A_i^{t-1}$ to constrain the model to focus on important information from previous model. Inclusive Distillation Loss overcomes catastrophic forgetting based on ABR.

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
