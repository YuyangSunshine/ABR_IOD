import os

import torch
import torch.utils.data
import torchvision
from PIL import Image
import sys
import scipy.io as scio
from scipy.io import loadmat

from maskrcnn_benchmark.data.datasets import COCODataset
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
import json
import cv2
import numpy
from maskrcnn_benchmark.data.transforms import Compose
from maskrcnn_benchmark.data.transforms.transforms import ToTensor

from maskrcnn_benchmark.structures.bounding_box import BoxList

def dict_slice(adict, start, end):
    keys = list(adict.keys())
    # print('keys : {0}'.format(keys))
    # print('length of keys: {0}'.format(len(keys)))
    dict_slice = {}
    for k in keys[start: end]:
        dict_slice[k] = adict[k]
    # print('dict_slice: {0}'.format(dict_slice))
    return dict_slice



def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    return False

def image_annotation(anno, classes):
    """
    only new categories' annotations
    """
    real_anno = []
    for i in anno:
        if PascalVOCDataset2012.CLASSES[i['category_id']] in classes:
            real_anno.append(i)
    return real_anno

def check_if_insert(anno,classes):
    for i in anno:
        if PascalVOCDataset2012.CLASSES[i['category_id']] in classes:
            return True

    return False



class PascalVOCDataset2012(torchvision.datasets.coco.CocoDetection):
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

    def __init__(self, data_dir, ann_file, split, use_difficult, transforms=None, external_proposal=False, old_classes=[],
                 new_classes=[], excluded_classes=[], is_train=True):
        super(PascalVOCDataset2012, self).__init__(data_dir, ann_file)
        self.ids = sorted(self.ids)
        self.is_train = is_train
        self.old_classes = old_classes
        self.new_classes = new_classes
        count = 0

        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                if self.is_train:
                    if check_if_insert(anno, new_classes):  # filtering images for new categories
                        count = count + 1
                        ids.append(img_id)
                else:
                    if check_if_insert(anno, new_classes+old_classes):  # filtering images for new categories
                        count = count + 1
                        ids.append(img_id)
        self.final_ids = ids
        if self.is_train:
            print('number of images used for training: {0}'.format(count))
        else:
            print('number of images used for testing: {0}'.format(count))
        self.num_img = count

        self.class_to_ind = dict(zip(PascalVOCDataset2012.CLASSES, range(len(PascalVOCDataset2012.CLASSES))))
        self._transforms = transforms

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index,shouldTransform=True):
        img, anno = super(PascalVOCDataset2012, self).__getitem__(index)
        id = self.final_ids[index]
        img = self._load_image(id)
        anno = self._load_target(id)
        proposal = None

        # filter annotation for old, new and exclude classes data
        if self.is_train:
            # print('before filtering, annotation: {0}'.format(anno))
            anno = image_annotation(anno, self.new_classes)
            # print('after filtering, annotation: {0}'.format(anno))
        else:
            anno = image_annotation(anno, self.new_classes + self.old_classes)
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = torch.Tensor([obj["category_id"] for obj in anno])
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='mask')
        for m in masks.instances:
            if (m==1).nonzero().shape[0] == 0:
                print()
        if len(masks.instances) == 0:
            print("something")
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=False)

        if self._transforms is not None and shouldTransform:
            proposal = None
            img, target, proposal = self._transforms(img, target, proposal)

        return img, target, proposal, index

    def __len__(self):
        return len(self.final_ids)

    def get_groundtruth(self, index):
        id = self.final_ids[index]
        img = self._load_image(id)
        # img = self.coco.loadImgs(id)[0]
        # img_size = [img["width"], img["height"]]
        anno = self._load_target(id)
        proposal = None

        # filter annotation for old, new and exclude classes data
        if self.is_train:
            # print('before filtering, annotation: {0}'.format(anno))
            anno = image_annotation(anno, self.new_classes)
            # print('after filtering, annotation: {0}'.format(anno))
        else:
            anno = image_annotation(anno, self.new_classes + self.old_classes)

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = torch.Tensor([obj["category_id"] for obj in anno])
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size, mode='mask')
        target.add_field("masks", masks)
        # id = self.final_ids[index]
        # anno = self._load_target(id)
        # anno = image_annotation(anno, self.new_classes + self.old_classes)
        # return anno
        # img_id = self.final_ids[index]
        # # anno = ET.parse(self._annopath % img_id).getroot()
        # # anno = self._preprocess_annotation(anno)
        # # height, width = anno["im_info"]
        # # self._img_height = height
        # # self._img_width = width
        # # target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        # # target.add_field("labels", anno["labels"])
        # # target.add_field("difficult", anno["difficult"])
        # anno_root = loadmat(self._annopath % img_id)
        # total_mask = torch.Tensor(anno_root["GTcls"][0]["Segmentation"][0])
        # classes = anno_root['GTcls'][0]["CategoriesPresent"][0]
        # if classes.shape[0] > 1:
        #     classes = classes.squeeze()
        # else:
        #     classes = classes[0]
        # masks = []
        # boxes = []
        # labels = []
        # h = total_mask.shape[0]
        # w = total_mask.shape[1]
        # for c in classes:
        #     if PascalVOCDataset2012.CLASSES[c] in self.new_classes:
        #         this_mask = total_mask.clone()
        #         this_mask[this_mask != c] = 0
        #         this_mask[this_mask == c] = 1
        #         mask_pixels = (this_mask == 1).nonzero(as_tuple=True)
        #         xmax = max(mask_pixels[1])
        #         ymax = max(mask_pixels[0])
        #         xmin = min(mask_pixels[1])
        #         ymin = min(mask_pixels[0])
        #         boxes.append(torch.tensor([xmin, ymin, xmax, ymax]))
        #         masks.append(this_mask)
        #         labels.append(c)
        #
        # masks = torch.stack(masks)
        # boxes = torch.stack(boxes)
        #
        #
        # target = BoxList(boxes, (w, h), mode="xyxy")
        # target.add_field("labels", torch.tensor(labels))
        # masks = SegmentationMask(masks, (w, h), mode='mask')
        # target.add_field("masks", masks)

        return target

    def get_proposal(self, index):
        boxes = []

        img_id = self.final_ids[index]
        proposal_path = self._proposalpath % "{0}".format(img_id)
        proposal_raw_data = scio.loadmat(proposal_path)
        proposal_data = proposal_raw_data['bbs']
        proposal_length = proposal_data.shape[0]
        for i in range(2000):
            # print('i: {0}'.format(i))
            if i >= proposal_length:
                break
            left = proposal_data[i][0]
            top = proposal_data[i][1]
            width = proposal_data[i][2]
            height = proposal_data[i][3]
            score = proposal_data[i][4]
            right = left + width
            bottom = top + height
            box = [left, top, right, bottom]
            boxes.append(box)
        img_height = self._img_height
        img_width = self._img_width

        boxes = torch.tensor(boxes, dtype=torch.float32)
        proposal = BoxList(boxes, (img_width, img_height), mode="xyxy")

        return proposal

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()

            old_class_flag = False
            for old in self.old_classes:
                if name == old:
                    old_class_flag = True
                    break
            exclude_class_flag = False
            for exclude in self.exclude_classes:
                if name == exclude:
                    exclude_class_flag = True
                    break

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [bb.find("xmin").text, bb.find("ymin").text, bb.find("xmax").text, bb.find("ymax").text]
            bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

            if exclude_class_flag:
                pass
                #print('voc.py | incremental train | object category belongs to exclude categoires: {0}'.format(name))
            elif self.is_train and old_class_flag:
                pass
                #print('voc.py | incremental train | object category belongs to old categoires: {0}'.format(name))
            else:
                boxes.append(bndbox)
                gt_classes.append(self.class_to_ind[name])
                difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        # img_id = self.final_ids[index]
        # size = loadmat(self._annopath % img_id)['GTcls'][0]["Segmentation"][0].shape
        # return {"height": size[0], "width": size[1]}
        img_id = self.final_ids[index]
        img_data = self.coco.imgs[img_id]
        return img_data


    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset2012.CLASSES[class_id]

    def get_img_id(self, index):
        img_id = self.final_ids[index]
        return img_id


def main():
    data_dir = "/home/DATA/VOC2007"
    split = "test"  # train, val, test
    use_difficult = False
    transforms = None
    dataset = PascalVOCDataset2012(data_dir, split, use_difficult, transforms)


if __name__ == '__main__':
    main()
