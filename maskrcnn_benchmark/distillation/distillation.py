import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import numpy as np

from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


def calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss=None, bbox_loss=None, bbox_threshold=None):

    rpn_objectness_source, rpn_bbox_regression_source = rpn_output_source
    rpn_objectness_target, rpn_bbox_regression_target = rpn_output_target

    # calculate rpn classification loss
    num_source_rpn_objectness = len(rpn_objectness_source)
    num_target_rpn_objectness = len(rpn_objectness_target)
    final_rpn_cls_distillation_loss = []
    objectness_difference = []

    if num_source_rpn_objectness == num_target_rpn_objectness:
        for i in range(num_target_rpn_objectness):
            current_source_rpn_objectness = rpn_objectness_source[i]
            current_target_rpn_objectness = rpn_objectness_target[i]
            if cls_loss == 'filtered_l2':
                rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                objectness_difference.append(rpn_objectness_difference)
                filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                rpn_difference = torch.max(rpn_objectness_difference, filter)
                rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for rpn classification distillation")
    else:
        raise ValueError("Wrong rpn objectness output")
    final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss)/num_source_rpn_objectness
    #a = objectness_difference > 0

    # calculate rpn bounding box regression loss
    num_source_rpn_bbox = len(rpn_bbox_regression_source)
    num_target_rpn_bbox = len(rpn_bbox_regression_target)
    final_rpn_bbs_distillation_loss = []
    l2_loss = nn.MSELoss(size_average=False, reduce=False)

    if num_source_rpn_bbox == num_target_rpn_bbox:
        for i in range(num_target_rpn_bbox):
            current_source_rpn_bbox = rpn_bbox_regression_source[i]
            current_target_rpn_bbox = rpn_bbox_regression_target[i]
            current_objectness_difference = objectness_difference[i]
            [N, A, H, W] = current_objectness_difference.size()  # second dimention contains location shifting information for each anchor
            current_objectness_difference = permute_and_flatten(current_objectness_difference, N, A, 1, H, W)
            current_source_rpn_bbox = permute_and_flatten(current_source_rpn_bbox, N, A, 4, H, W)
            current_target_rpn_bbox = permute_and_flatten(current_target_rpn_bbox, N, A, 4, H, W)
            current_objectness_mask = current_objectness_difference.clone()
            current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
            current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
            masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
            masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
            if bbox_loss == 'l2':
                current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
                final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
            elif bbox_loss == 'None':
                final_rpn_bbs_distillation_loss.append(0)
            else:
                raise ValueError('Wrong loss function for rpn bounding box regression distillation')
    else:
        raise ValueError('Wrong RPN bounding box regression output')
    final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/num_source_rpn_bbox

    final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
    final_rpn_loss.to('cuda')

    return final_rpn_loss


def calculate_attentive_roi_feature_distillation(f_map_s, f_map_t, gamma=1.0):
    """
    Args:
        f_map_s(Tensor): Bs*C*H*W, student's feature map
        f_map_t(Tensor): Bs*C*H*W, teacher's feature map
    """
    temp = 2

    S_attention_t = activation_at(f_map_s, temp)
    S_attention_s = activation_at(f_map_t, temp)

    loss_pad = pad_loss(S_attention_s, S_attention_t)
    loss_afd = afd_loss(f_map_s, f_map_t, S_attention_t)
    combined_loss = loss_afd + gamma*loss_pad
    return combined_loss
    

def afd_loss(f_map_s, f_map_t, S_t):
    loss_mse = nn.MSELoss(reduction='mean')
    S_t = S_t.unsqueeze(dim=1)
 
    fea_t = torch.mul(f_map_t, torch.sqrt(S_t))
    fea_s = torch.mul(f_map_s, torch.sqrt(S_t))
    
    loss = loss_mse(fea_s, fea_t)
    return loss


def pad_loss(S_s, S_t):
    loss_l1 = nn.L1Loss(reduction='mean')
    loss = loss_l1(S_s, S_t)

    return loss


def activation_at(f_map, temp=2):
    N, C, H, W = f_map.shape

    value = torch.abs(f_map)
    # Bs*W*H
    fea_map = value.pow(temp).mean(axis=1, keepdim=True)
    # S_attention = (H * W * F.softmax(fea_map.view(N, -1)/temp, dim=1)).view(N, H, W)
    S_attention = (H * W * F.softmax(fea_map.view(N, -1), dim=1)).view(N, H, W)
    
    return S_attention


def calculate_feature_distillation_loss(source_features, target_features, loss=None):  # pixel-wise
    num_source_features = len(source_features)
    num_target_features = len(target_features)
    final_feature_distillation_loss = []

    if num_source_features == num_target_features:
        for i in range(num_source_features):
            source_feature = source_features[i]
            target_feature = target_features[i]
            if loss == 'normalized_filtered_l1':
                source_feature_avg = torch.mean(source_feature)
                target_feature_avg = torch.mean(target_feature)
                normalized_source_feature = source_feature - source_feature_avg  # normalize features
                normalized_target_feature = target_feature - target_feature_avg
                feature_difference = normalized_source_feature - normalized_target_feature
                feature_size = feature_difference.size()
                filter = torch.zeros(feature_size).to('cuda')
                feature_distillation_loss = torch.max(feature_difference, filter)
                final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                del filter
                torch.cuda.empty_cache()  # Release unoccupied memory
            else:
                raise ValueError("Wrong loss function for feature distillation")
    else:
        raise ValueError("Number of source features must equal to number of target features")

    final_feature_distillation_loss = sum(final_feature_distillation_loss)

    return final_feature_distillation_loss


def calculate_roi_distillation_loss(soften_results, target_results, cls_preprocess=None, cls_loss=None, bbs_loss=None, temperature=1, soften_proposal=None):

    soften_scores, soften_bboxes = soften_results
    target_scores, target_bboxes = target_results
    num_of_distillation_categories = soften_scores.size()[1]

    # compute distillation loss
    if cls_preprocess == 'normalization':
        class_wise_soften_scores_avg = torch.mean(soften_scores, dim=1).view(-1, 1)
        class_wise_target_scores_avg = torch.mean(target_scores, dim=1).view(-1, 1)
        normalized_soften_scores = torch.sub(soften_scores, class_wise_soften_scores_avg)
        normalized_target_scores = torch.sub(target_scores, class_wise_target_scores_avg)
        modified_soften_scores = normalized_target_scores[:, : num_of_distillation_categories]  # include background
        modified_target_scores = normalized_soften_scores[:, : num_of_distillation_categories]  # include background
    elif cls_preprocess == 'inclusive_distillation':  # FOR UNBIAS CROSS ENTROPY USE THIS
        modified_soften_scores = soften_scores
        modified_target_scores = target_scores
    else:
        raise ValueError("Wrong preprocessing method for raw classification output")

    tot_classes = target_scores.size()[1]
    if cls_loss == 'l2':
        l2_loss = nn.MSELoss(size_average=False, reduce=False)
        class_distillation_loss = l2_loss(modified_soften_scores, modified_target_scores)
        class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
    elif cls_loss == 'unbiased-cross-entropy':
        # align the probobilities of the source model for the background class
        # with the probobilities of the target model for both background class([0]) and current classes(num_of_distillation_categories)
        new_bkg_idx = torch.tensor([0] + [x for x in range(
            num_of_distillation_categories, tot_classes)]).to(target_scores.device)
        den = torch.logsumexp(modified_target_scores, dim=1)
        outputs_no_bgk = modified_target_scores[:, 1:-(tot_classes-num_of_distillation_categories)] - den.unsqueeze(dim=1)
        outputs_bkg = torch.logsumexp(torch.index_select(modified_target_scores, index=new_bkg_idx, dim=1), dim=1) - den
        labels = torch.softmax(modified_soften_scores, dim=1)
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / soften_scores.shape[1]
        class_distillation_loss = -torch.mean(loss)
    else:
        raise ValueError("Wrong loss function for classification")

    # compute distillation bbox loss
    modified_soften_boxes = soften_bboxes[:, 1:, :]  # exclude background bbox
    modified_target_bboxes = target_bboxes[:, 1:num_of_distillation_categories, :]  # exclude background bbox
    if bbs_loss == 'l2':
        l2_loss = nn.MSELoss(size_average=False, reduce=False)
        bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_soften_boxes)
        bbox_distillation_loss = torch.mean(torch.mean(torch.sum(bbox_distillation_loss, dim=2), dim=1), dim=0)  # average towards categories and proposals
    elif bbs_loss == 'smooth_l1':
        num_bboxes = modified_target_bboxes.size()[0]
        num_categories = modified_target_bboxes.size()[1]
        bbox_distillation_loss = smooth_l1_loss(modified_target_bboxes, modified_soften_boxes, size_average=False, beta=1)
        bbox_distillation_loss = bbox_distillation_loss / (num_bboxes * num_categories)  # average towards categories and proposals
    else:
        raise ValueError("Wrong loss function for bounding box regression")

    roi_distillation_losses = torch.add(class_distillation_loss, bbox_distillation_loss)

    return roi_distillation_losses


def calculate_roi_distillation_losses(soften_results, target_results, dist='l2', soften_proposal=None):

    if dist == 'id':
        if soften_proposal == None:
            cls_preprocess = 'inclusive_distillation'
        cls_loss = 'unbiased-cross-entropy'
        bbs_loss = 'l2'
        temperature = 1
    else:
        cls_preprocess = 'normalization'
        cls_loss = 'l2'
        bbs_loss = 'l2'
        temperature = 1

    roi_distillation_losses = calculate_roi_distillation_loss(
        soften_results, target_results, cls_preprocess, cls_loss, bbs_loss, temperature, soften_proposal)

    return roi_distillation_losses


