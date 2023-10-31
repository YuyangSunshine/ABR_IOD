# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import torch
import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from tqdm import tqdm


def do_voc_evaluation_inst(dataset, predictions, output_folder, logger):
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    iou_thresholds = np.arange(0.5, 0.95, 0.05).tolist()
    ap_boxes = np.zeros((len(iou_thresholds), len(dataset.new_classes) + len(dataset.old_classes)))
    ap_masks = np.zeros((len(iou_thresholds), len(dataset.new_classes) + len(dataset.old_classes)))

    for idx, iou_tresh in tqdm(enumerate(iou_thresholds), total=len(iou_thresholds)):
        result = eval_detection_voc(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=iou_tresh,
            use_07_metric=False,
        )
        ap_masks[idx] = result['ap_mask'][1:]
        ap_boxes[idx] = result['ap_box'][1:]

    ap_05_95_mask = ap_masks.mean(axis=0)
    ap_05_95_box = ap_boxes.mean(axis=0)
    result_str_box = "mAP OD\n {:.4f}\n".format(np.mean(ap_05_95_box))
    result_str_mask = "mAP IS\n {:.4f}\n".format(np.mean(ap_05_95_mask))

    for i, ap in enumerate(ap_05_95_box):
        result_str_box += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i+1), ap
        )
    for i, ap in enumerate(ap_05_95_mask):
        result_str_mask += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i+1), ap
        )
    print("BOX", end=": ")
    print(",".join([str(x) for x in ap_05_95_box]))
    print("MSK", end=": ")
    print(",".join([str(x) for x in ap_05_95_mask]))

    logger.info(result_str_box)
    logger.info(result_str_mask)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str_box)
            fid.write(result_str_mask)

    return {"mask": ap_05_95_mask, "box": result_str_box}


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    prec, rec, mask_prec, mask_rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap_box = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    ap_mask = calc_detection_voc_ap(mask_prec, mask_rec, use_07_metric=use_07_metric)
    return {"ap_box": ap_box, "ap_mask": ap_mask, "map_box": np.nanmean(ap_box), "map_mask": np.nanmean(ap_mask)}

def masklist_iou(mask_target, mask_predicted):
    n_gt_masks = mask_target.shape[0]
    n_pred_masks = mask_predicted.shape[0]
    mask_target = torch.Tensor(mask_target)
    mask_predicted = torch.Tensor(mask_predicted)
    ious = np.zeros((n_pred_masks, n_gt_masks))
    for p in range(n_pred_masks):
        for t in range(n_gt_masks):
            tp_px = ((mask_target[t] - mask_predicted[p])[mask_target[t] == 1] == 0).nonzero().size(0)
            fp_px = ((mask_target[t] - mask_predicted[p])[mask_target[t] == 0] == -1).nonzero().size(0)
            fn_px = ((mask_target[t] - mask_predicted[p])[mask_target[t] == 1] == 1).nonzero().size(0)
            if (tp_px+fp_px+fn_px) == 0:
                ious[p][t] = 0.0
                break
            IoU = tp_px/(tp_px+fp_px+fn_px)
            ious[p][t] = IoU
    return ious

def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    mask_match = defaultdict(list)
    for gt_boxlist, pred_boxlist in tqdm(zip(gt_boxlists, pred_boxlists), total=len(gt_boxlists)):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        pred_segmask = pred_boxlist.get_field("mask").instances.masks.cpu().numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_segmask = gt_boxlist.get_field("masks").instances.masks.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            pred_segmask_l = pred_segmask[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_segmask_l = pred_segmask_l[order]
            pred_score_l = pred_score_l[order]
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_segmask_l = gt_segmask[gt_mask_l]
            n_pos[l] += gt_bbox_l.shape[0]
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                mask_match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            box_iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            mask_iou = masklist_iou(gt_segmask_l, pred_segmask_l)
            box_gt_index = box_iou.argmax(axis=1)
            mask_gt_index = mask_iou.argmax(axis=1)
            # set -1 if there is no matching ground truth

            box_gt_index[box_iou.max(axis=1) < iou_thresh] = -1
            mask_gt_index[mask_iou.max(axis=1) < iou_thresh] = -1

            del box_iou
            del mask_iou
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for box_gt_idx in box_gt_index:
                if box_gt_idx >= 0:
                    if not selec[box_gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[box_gt_idx] = True
                else:
                    match[l].append(0)

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for mask_gt_idx in mask_gt_index:
                if mask_gt_idx >= 0:
                    if not selec[mask_gt_idx]:
                        mask_match[l].append(1)
                    else:
                        mask_match[l].append(0)
                    selec[mask_gt_idx] = True
                else:
                    mask_match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    mask_prec = [None] * n_fg_class
    mask_rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        mask_match_l = np.array(mask_match[l], dtype=np.int8)
        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        mask_match_l = mask_match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)
        mask_tp = np.cumsum(mask_match_l == 1)
        mask_fp = np.cumsum(mask_match_l == 0)

        prec[l] = tp / (fp + tp)
        mask_prec[l] = mask_tp / (mask_fp + mask_tp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            mask_rec[l] = mask_tp / n_pos[l]


    return prec, rec, mask_prec, mask_rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
