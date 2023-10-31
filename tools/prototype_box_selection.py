# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from ast import Str
from email.mime import image
from sys import setprofile
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from maskrcnn_benchmark.config import \
    cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from maskrcnn_benchmark.data.build import make_bbox_loader # import data set
from maskrcnn_benchmark.engine.inference import inference  # inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict  # when multiple gpus are used, reduce the loss
from maskrcnn_benchmark.modeling.detector import build_detection_model  # used to create model
from maskrcnn_benchmark.solver import make_lr_scheduler  # learning rate updating strategy
from maskrcnn_benchmark.solver import make_optimizer  # setting the optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank  # related to multi-gpu training; when usong 1 gpu, get_rank() will return 0
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger  # related to logging model(output training status)
from maskrcnn_benchmark.utils.miscellaneous import mkdir  # related to folder creation
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
# from torch.utils.tensorboard import SummaryWriter
from maskrcnn_benchmark.distillation.distillation import calculate_rpn_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_feature_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_roi_distillation_losses
import random

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

import warnings

import os
import pickle
from tools.extract_memory import Mem


warnings.filterwarnings("ignore", category=UserWarning)


def extract_bboxes_and_features(model_source, data_loader, device, cfg):

    old_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES
    new_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES
    
    # record log information
    logger = logging.getLogger("maskrcnn_benchmark_last_model.trainer")
    logger.info("Start sampling")
    meters = MetricLogger(delimiter="  ")  # used to record

    #################################################################
	# extract the feature maps for each boxes per classes in images #
	#################################################################

    # Start extracting!
    max_iter = len(data_loader)  # data loader rewrites the len() function and allows it to return the number of batches (cfg.SOLVER.MAX_ITER)
    data_iter = iter(data_loader)
    model_source.eval()  # set the source model in inference mode
    start_training_time = time.time()
    end = time.time()

    all_bboxes_info = [[] for _ in range(len(new_classes))]

    for i in range(len(data_loader)):
        try:
            images, targets, original_targets, idx = next(data_iter)
        except StopIteration:
            break
        # print("Current load data is {0}/{1}".format(i, len(data_loader)))
        data_time = time.time() - end

        images = images.to(device)  # move images to the device
        targets = [target.to(device) for target in targets]  # move targets (labels) to the device

        # extract the features for each rois
        with torch.no_grad():
            (target_scores, _), _, _, _, roi_align_features = \
                model_source.generate_feature_logits_by_targets(images, targets)

        target_scores.tolist() # [9, 16]
        roi_align_features = torch.mean(roi_align_features.cpu(), dim=1).tolist() # [9, 1024, 7, 7]
        
        # print(target_scores.shape)
        # time used to do one batch processing
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        bbox_index = 0

        for img_n in range(len(idx)):

            target = original_targets[img_n]
            
            ######## visulation ########
            # img_id = ids_[img_n][0]
            # img = Image.open("data/voc07/VOCdevkit/VOC2007/JPEGImages/{0}.jpg".format(img_id)).convert("RGB")
            # from PIL import ImageDraw
            # from PIL import ImageFont
            # a = ImageDraw.ImageDraw(img)
            # ttf = ImageFont.load_default()
            # for g in range(target.__len__()):
            #     gt_ = target.bbox.tolist()[g]
            #     label_ = target.extra_fields['labels'].tolist()[g]
            #     a.rectangle(((gt_[0], gt_[1]), (gt_[2], gt_[3])), fill=None, outline='blue', width=4)
            #     a.text((gt_[0]+5, gt_[1]+6), str(label_), font=ttf, fill=(0,0,255))
            # img.save('output/box_rehearsal/current_image_{}.jpg'.format(img_id))

            for ind in range(target.__len__()):
                bboxes = target.bbox[ind].cpu().tolist()
                
                bbox_index+=1
                # Delete too small boxes.
                if (bboxes[2]-bboxes[0]) <= 70 and (bboxes[3]-bboxes[1]) <= 70:
                    continue
                else:
                    # print(len(roi_align_features))
                    # print(bbox_index-1)
                    all_bboxes_info[target.extra_fields["labels"][ind].item()-len(old_classes) - 1].append(
                        {'feature': roi_align_features[bbox_index-1], # [1024, 7, 7]
                        'logits': target_scores[img_n+ind].cpu(), # [16]
                        'image_path': idx[img_n], # list
                        'box_class': target.extra_fields["labels"][ind].cpu().item(), # []
                        'box': bboxes, # []
                        'mode': target.mode}) # str

    # Saving the old bbox and image information through the frozen model
    # Under the default continuous learning settings, all information should not be saved and is only used for debugging！！
    # with open(old_bbox_information_file, 'wb') as obf:
    #     pickle.dump(all_bboxes_info, obf, pickle.HIGHEST_PROTOCOL)
    #     print("The old_bbox_information size is: {}".format(len(all_bboxes_info)))
    #     print("All_information is saved in: {}".format(old_bbox_information_file))

    # Display the total used training time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total sampling time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

    return all_bboxes_info

def selector(cfg_source, num_gpus):
    current_mem_file = f"{cfg_source.MEM_TYPE}_{cfg_source.MEM_BUFF}"

    # creat or load the memory file
    # if cfg_source.STEP == 0:
    #     # Creat a memory buffer for first task (first task is the same for different setting)
    #     current_mem_path = os.path.split(cfg_source.MODEL.SOURCE_WEIGHT)[0] + current_mem_file
    #     if not os.path.exists(current_mem_path):
    #         os.mkdir(current_mem_path)
    # else:
    #     # Create the corresponding memory buffer for incremental steps
    current_mem_path = os.path.join(cfg_source.OUTPUT_DIR, current_mem_file)
    if not os.path.exists(current_mem_path):
        os.mkdir(current_mem_path)
    
    print('-- PBS REPORT-- The current Box Reahersal path is {0}'.format(current_mem_path))

    # Select the prototype box image
    num_file_classes = len(os.listdir(current_mem_path))

    if cfg_source.STEP == 0 and num_file_classes>=int(cfg_source.MEM_BUFF):
        # The prototype boxes have exsisted for current step!
        print("The prototype box images for first step have existed!!")
        all_bboxes_info = None
    else: # Update the prototype boxes for current classes
        model_source = build_detection_model(cfg_source)  # create the source model
        device = torch.device(cfg_source.MODEL.DEVICE)  # default is "cuda"
        model_source.to(device)  # move source model to gpu
        
        arguments_source = {}
        arguments_source["iteration"] = 0
        # path to store the trained parameter value
        output_dir_source = cfg_source.OUTPUT_DIR + f"STEP{cfg_source.STEP}"

        # create check pointer for source model & load the pre-trained model parameter to source model
        checkpointer_source = DetectronCheckpointer(cfg_source, model_source, save_dir=output_dir_source)
        extra_checkpoint_data_source = checkpointer_source.load(cfg_source.MODEL.WEIGHT)
        arguments_source.update(extra_checkpoint_data_source)

        # load training data
        bbox_loader = make_bbox_loader(cfg_source, is_train=False, num_gpus=num_gpus, rank=get_rank())

        # get the memory from the model
        all_bboxes_info = extract_bboxes_and_features(model_source, bbox_loader, device, cfg_source)

    ##############################################################################
	# create or update memory
	##############################################################################

    Exemplar = Mem(cfg_source, cfg_source.STEP, current_mem_path)
    Exemplar.update_memory(all_bboxes_info)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--rpn",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--feat",
        default="no",
        type=str, choices=['no', 'std', 'align', 'att']
    )
    parser.add_argument(
        "--uce",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--init",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--inv",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--mask",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--cls",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="10-10"
    )
    parser.add_argument(
        "-n", "--name",
        default="LR005_BS4_FILOD",
    )
    parser.add_argument(
        "-s", "--step",
        default=0, type=int,
    )
    parser.add_argument(
        "-cvd", "--cuda_visible_devices",
        type=str,
        help="Select the specific GPUs",
        default="0"
    )
    parser.add_argument(
        "-mb", "--memory_buffer",
        default=2000, type=int,
    )
    parser.add_argument(
        "-mt", "--memory_type",
        default="mean", type=str,
        choices=['random', 'herding', 'mean']
    )
    parser.add_argument(
        "-iss", "--is_sample",
        default=True,
        action='store_true',
    )

    args = parser.parse_args()
    # setting the corresponding GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.step == 0:
        source_model_config_file = f"configs/voc/{args.task}/e2e_faster_rcnn_R_50_C4_4x.yaml"
    else:
        source_model_config_file = f"configs/voc/{args.task}/e2e_faster_rcnn_R_50_C4_4x_RB_Target_model.yaml"
    full_name = f"{args.name}/"  # if args.step > 1 else args.name

    if args.local_rank != 0:
        return
    num_gpus = 1

    cfg_source = cfg.clone()
    cfg_source.merge_from_file(source_model_config_file)

    base = 'output'
    # setting the weight for source and target model
    if args.step == 0:
        cfg_source.MODEL.SOURCE_WEIGHT = f"{cfg_source.OUTPUT_DIR}/model_trimmed.pth"
        cfg_source.MODEL.WEIGHT = cfg_source.MODEL.SOURCE_WEIGHT
        # print(cfg_source.MODEL.SOURCE_WEIGHT)
    elif args.step >= 1:
        cfg_source.MODEL.WEIGHT = f"{base}/{args.task}/{args.name}/STEP{args.step}/model_trimmed.pth"
        cfg_source.OUTPUT_DIR = f"{base}/{args.task}/{args.name}"
        if cfg_source.OUTPUT_DIR:
            mkdir(cfg_source.OUTPUT_DIR)

    # setting the output head 
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES += args.step * cfg_source.CLS_PER_STEP
        cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES += cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[:(args.step - 1) * cfg_source.CLS_PER_STEP]
        print(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
        cfg_source.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES = cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[args.step * cfg_source.CLS_PER_STEP:]
        print(cfg_source.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES)
        cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES = cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[(args.step - 1) * cfg_source.CLS_PER_STEP: args.step * cfg_source.CLS_PER_STEP]
        print(cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)
    else:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES) + 1

    cfg_source.TASK = args.task
    cfg_source.STEP = args.step
    cfg_source.NAME = args.name
    cfg_source.IS_FATHER = False
    cfg_source.IS_SAMPLE = args.is_sample
    cfg_source.TEST.IMS_PER_BATCH = 8

    cfg_source.MEM_BUFF = args.memory_buffer
    cfg_source.MEM_TYPE = args.memory_type
    cfg_source.freeze()

    # use current model to select the prototype boxes
    selector(cfg_source, num_gpus)


if __name__ == "__main__":
    main()
