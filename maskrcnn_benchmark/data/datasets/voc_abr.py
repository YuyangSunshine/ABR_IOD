from email.mime import image
from operator import gt
import os
from re import T
from tracemalloc import is_tracing

import torch
import torch.utils.data
from PIL import Image, ImageFilter
import sys
import scipy.io as scio
import cv2
import numpy as np
import random
from maskrcnn_benchmark.data.transforms import Compose

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList
from tools.extract_memory import Mem
import math


class PascalVOCDataset(torch.utils.data.Dataset):
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, external_proposal=False, old_classes=[],
                 new_classes=[], excluded_classes=[], is_train=True, is_father=True, cfg=None):
        self.cfg_ = cfg
        self.root = data_dir
        self.image_set = split  # train, validation, test
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.use_external_proposal = external_proposal

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        self._proposalpath = os.path.join(self.root, "EdgeBoxesProposals", "%s.mat")

        self._img_height = 0
        self._img_width = 0

        self.old_classes = old_classes
        self.new_classes = new_classes
        self.exclude_classes = excluded_classes
        
        self.is_train = is_train
        self.is_father = is_father

        self.final_ids = []
        
        # load data from all categories
        # self._normally_load_voc()

        if self.is_father:
            # do not use old data
            if self.is_train:
                print('voc.py | in training mode') # training mode 
                self._load_img_from_NEW_cls_without_old_data()
            elif not self.is_train:
                print('voc.py | in test mode')
                self._load_img_from_NEW_and_OLD_cls_without_old_data()

    def _normally_load_voc(self):
        """ load data from all 20 categories """

        # print("voc.py | normally_load_voc | load data from all 20 categories")
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.final_ids = self.ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}  # image_index : image_id

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))  # class_name : class_id

    def _load_img_from_NEW_and_OLD_cls_without_old_data(self):
        self.ids = []
        total_classes = self.new_classes + self.old_classes
        for w in range(len(total_classes)):
            category = total_classes[w]
            img_per_categories = []
            with open(self._imgsetpath % "{0}_{1}".format(category, self.image_set)) as f:
                buff = f.readlines()
            buff = [x.strip("\n") for x in buff]
            for i in range(len(buff)):
                a = buff[i]
                b = a.split(' ')
                if b[1] == "-1":  # do not contain the category object
                    pass
                elif b[2] == '0':  # contain the category object -- difficult level
                    if self.is_train:
                        pass
                    else:
                        img_per_categories.append(b[0])
                        self.ids.append(b[0])
                else:
                    img_per_categories.append(b[0])
                    self.ids.append(b[0])
            # print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | number of images in {0}_{1}: {2}'.format(category, self.image_set, len(img_per_categories)))

        # check for image ids repeating
        self.final_ids = []
        for id in self.ids:
            repeat_flag = False
            for final_id in self.final_ids:
                if id == final_id:
                    repeat_flag = True
                    break
            if not repeat_flag:
                self.final_ids.append(id)
        # print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        
    def _load_img_from_NEW_cls_without_old_data(self):
        self.ids = []
        for incremental in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                x = buff[i]
                x = x.split(' ')
                if x[1] == '-1':
                    pass
                elif x[2] == '0':  # include difficult level object
                    if self.is_train:
                        pass
                    else:
                        img_ids_per_category.append(x[0])
                        self.ids.append(x[0])
                else:
                    img_ids_per_category.append(x[0])
                    self.ids.append(x[0])
            # print('voc.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)
            # print('voc.py | load_img_from_NEW_and_OLD_cls_without_old_data | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # print(self.final_ids)
        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.final_ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.use_external_proposal:
            proposal = self.get_proposal(index)
            proposal = proposal.clip_to_image(remove_empty=True)
        else:
            proposal = None

        if self.transforms is not None:
            img, target, proposal = self.transforms(img, target, proposal)

        return img, target, proposal, index

    def __len__(self):
        return len(self.final_ids)

    def get_groundtruth(self, index):
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        self._img_height = height
        self._img_width = width
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
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
        
        # normal for train or test
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
        img_id = self.final_ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]

    def get_img_id(self, index):
        img_id = self.final_ids[index]
        return img_id


class PascalVOCDataset_ABR(PascalVOCDataset):
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")
    """
    CLASSES = ("__background__ ", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor")

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, external_proposal=False, old_classes=[],
                 new_classes=[], excluded_classes=[], is_train=True, cfg=None):
        """
            Augmented Box Replay for PascalVOCDataset.
        """
        self.cfg = cfg  # import the basic cfg for setting the dataset
        self.is_father = self.cfg.IS_FATHER # for general dataloader

        # set the father Class
        super().__init__(data_dir, split, use_difficult, transforms, external_proposal, old_classes, new_classes, excluded_classes, is_train, self.is_father, cfg=cfg)

        if not self.is_father:
            self.total_classes = self.old_classes + self.new_classes
            self.total_num_classes = len(self.total_classes)
            self.batch_size = cfg.SOLVER.IMS_PER_BATCH

            ### ---- Memory Managment Configation -----###
            self.is_mem = False
            if cfg.MEM_BUFF != None: # start to use the rehearsal (box images)
                self.is_mem = True
            self.mem_buff=self.cfg.MEM_BUFF # memory buffer size
            self.mem_type=self.cfg.MEM_TYPE # the type for choosing memory
            self.is_sample = cfg.IS_SAMPLE # is the sampling phase or not

            self.PrototypeBoxSelection = None
            self.BoxRehearsal_path = None # the box rehearsal memory
            self.boxes_index = [] # the box-memory index
            self.bg_size = 0

            if self.is_train and self.is_mem:  # training mode
                print('voc.py | in training with box rehearsal memory mode')
                self._load_img_from_NEW_and_OLD_cls_with_old_mem()
            elif not self.is_train and self.is_sample: # sampling mode
                print('voc.py | in sampling mode')
                self._load_img_from_NEW_cls()
            elif not self.is_train and not self.is_sample: # testing mode
                print('voc.py | in test mode')
                self._load_img_from_NEW_and_OLD_cls_without_old_data()

    def _load_img_from_NEW_and_OLD_cls_with_old_mem(self):
        """ load new data with only new classes, and box rehearsal memory with only old classes """
        cls = PascalVOCDataset_ABR.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        
        ###### 1. loading the current images with new classes  ######
        self.new_ids = [] # only the current images index number
        # only for the new classes images
        for incremental in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)) as f:
                print("TXT FILE {0}".format(self._imgsetpath % "{0}_{1}".format(incremental, self.image_set)))
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                x = buff[i]
                x = x.split(' ')
                # -1 without this class
                if x[1] == '-1':
                    pass
                # 0 difficult to detect but contain
                elif x[2] == '0':  # include difficult level object
                    if self.is_train:
                        pass
                    else:
                        img_ids_per_category.append(x[0])
                        self.new_ids.append(x[0])
                else:
                # 1 contains this class
                    img_ids_per_category.append(x[0])
                    self.new_ids.append(x[0])
            # print('voc.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

        # check for new image ids repeating
        self.final_ids = []
        for id in self.new_ids:
            repeat_flag = False
            for final_id in self.final_ids:
                if id == final_id:
                    repeat_flag = True
                    break
            if not repeat_flag:
                self.final_ids.append(id)
        print('voc.py | load_img_from_NEW_cls | total used number of new images in {0}: {1}'.format(self.image_set, len(self.final_ids)))
        
        ###### 2. loading box rehearsal memory images of task task t-1 ######
        
        self.PrototypeBoxSelection = Mem(self.cfg, self.cfg.STEP)
        self.BoxRehearsal_path = self.PrototypeBoxSelection.exemplar
        random.shuffle(self.BoxRehearsal_path)
        
        self.boxes_index = list(range(len(self.BoxRehearsal_path)))

        # print('voc.py | load_img_from_NEW_and_OLD_cls_with_old_mem | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))
        print('voc.py | load_boxes_from_old_mem | total used number of boexes: {0}'.format(len(self.BoxRehearsal_path)))
        
        # store new image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        
    def _load_img_from_NEW_cls(self):
        """ Just for sampling! load the current images (new classes) """
        self.ids = []
        for new in self.new_classes:  # read corresponding class images from the data set
            img_ids_per_category = []
            with open(self._imgsetpath % "{0}_{1}".format(new, self.image_set)) as f:
                buff = f.readlines()
                buff = [x.strip("\n") for x in buff]

            for i in range(len(buff)):
                x = buff[i]
                x = x.split(' ')
                # -1 without this class
                if x[1] == '-1':
                    pass
                # 0 difficult to detect but contain
                elif x[2] == '0':  # include difficult level object
                    pass
                else:
                # 1 contains this class
                    img_ids_per_category.append(x[0])
                    self.ids.append(x[0])

            # print('voc.py | load_img_from_NEW_cls_without_old_data | number of images in {0}_{1} set: {2}'.format(incremental, self.image_set, len(img_ids_per_category)))

            # check for image ids repeating
            self.final_ids = []
            for id in self.ids:
                repeat_flag = False
                for final_id in self.final_ids:
                    if id == final_id:
                        repeat_flag = True
                        break
                if not repeat_flag:
                    self.final_ids.append(id)
            # print('voc.py | load_img_from_NEW_cls_for_sampling | total used number of images in {0}: {1}'.format(self.image_set, len(self.final_ids)))

        # store image ids and class ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.final_ids)}
        cls = PascalVOCDataset_ABR.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        if self.is_train:
            if self.is_father:
                img_id = self.final_ids[index]
                img = Image.open(self._imgpath % img_id).convert("RGB")

                target = self.get_groundtruth(index)
                target = target.clip_to_image(remove_empty=True)

                if self.use_external_proposal:
                    proposal = self.get_proposal(index)
                    proposal = proposal.clip_to_image(remove_empty=True)
                else:
                    proposal = None

                if self.transforms is not None:
                    img, target, proposal = self.transforms(img, target, proposal)

                return img, target, proposal, index
            else:
                ##### load current image and annotation for training #####
                img_id = self.final_ids[index]
                img = Image.open(self._imgpath % img_id).convert("RGB")
                # img.save('output/imgorign_{}.jpg'.format(index))

                target = self.get_groundtruth_ABR(index)
                target = target.clip_to_image(remove_empty=True)

                if self.use_external_proposal:
                    proposal = self.get_proposal(index)
                    proposal = proposal.clip_to_image(remove_empty=True)
                else:
                    proposal = None

                (current_image, current_targets) = (img, target)

                ### Start transform_current_data_with_ABR ###
                _current_image, _current_targets = self.transform_current_data_with_ABR(current_image, current_targets)

                if self.transforms is not None:
                    _current_image, _current_targets, proposal = self.transforms(_current_image, _current_targets, proposal)

                return _current_image, _current_targets, proposal, img_id
        else:
            #### load images for sampling ####

            img_id = self.final_ids[index]
            img = Image.open(self._imgpath % img_id).convert("RGB")

            target = self.get_groundtruth_ABR(index)
            target = target.clip_to_image(remove_empty=True)

            proposal = None
            original_target = target

            if self.transforms is not None:
                img, target, _ = self.transforms(img, target, proposal)

            return img, target, original_target, [img_id]

    def __len__(self):
        return len(self.final_ids)

    def _sample_per_bbox_from_boxrehearsal(self, i, im_shape):
        """Sample a box from the BoxRehearsal.
        Args:
            i (int): Index of the BoxRehearsal.
            im_shape (tuple): Shape of the current image.
        Returns:
            Image: The box image.
            np.ndarray: Cropped ground truth boxes.
            int: Index of the sampled box.
        """
        # Get the path to the box image
        box_im_path = self.PrototypeBoxSelection.current_mem_path
        if self.PrototypeBoxSelection.current_mem_path==None:
            box_im_path = self.PrototypeBoxSelection.first_mem_path

        box_im_path = os.path.join(box_im_path, self.BoxRehearsal_path[self.boxes_index[i]])

        # Open the box image and extract class name and index
        box_im = Image.open(box_im_path).convert("RGB")
        cls_name, index = os.path.splitext(self.BoxRehearsal_path[self.boxes_index[i]])[0].split('_')

        bboxes = [0, 0, box_im.size[0], box_im.size[1]]
        box_o_h, box_o_w = box_im.size[1], box_im.size[0]
        gt_classes = int(cls_name)

        # Calculate mean size of current input image and box
        im_mean_size = np.mean(im_shape)
        box_mean_size = np.mean(np.array([int(bboxes[2]), int(bboxes[3])]))
        
         # Modify the box size based on mean sizes
        if float(box_mean_size) >= float(im_mean_size*0.2) and float(box_mean_size) <= float(im_mean_size*0.7):
            box_scale = 1.0
        else:
            box_scale = random.uniform(float(im_mean_size*0.4), float(im_mean_size*0.6)) / float(box_mean_size)
        
        # Resize the box image
        box_im = box_im.resize((int(box_scale * box_o_w), int(box_scale * box_o_h)))
        
        # Define ground truth boxes
        gt_boxes = [0, 0, box_im.size[0], box_im.size[1], gt_classes]
        
        return box_im, np.array([gt_boxes]), self.boxes_index[i]
    
    def _start_mixup(self, image, targets, alpha=2.0, beta=5.0):
        """ Mixup the input image

        Args:
            image : the original image
            targets : the original image's targets
        Returns:
            mixupped images and targets
        """
        image = np.array(image)
        # image.flags.writeable = True
        img_shape = image.shape
        
        if not isinstance(targets, np.ndarray):
            gts = []
            bbox_list = targets.bbox.tolist()
            label_list = targets.extra_fields["labels"].tolist()
            for i in range(len(bbox_list)):
                gts.append(bbox_list[i] + [label_list[i]])
            gts = np.array(gts)
        else:
            gts = targets
            
        # make sure the image has more than one targets
        # If the only target occupies 75% of the image, we abandon mixupping.
        _MIXUP=True
        if gts.shape[0] == 1:
            img_w = gts[0][2]-gts[0][0]
            img_h = gts[0][3]-gts[0][1]
            if (img_shape[1]-img_w)<(img_shape[1]*0.25) and (img_shape[0]-img_h)<(img_shape[0]*0.25):
                _MIXUP=False
        
        ##### For normal mixup ######
        if _MIXUP: # 
            # lambda: Sampling from a beta distribution 
            Lambda = torch.distributions.beta.Beta(alpha, beta).sample().item()
            num_mixup = 3 # more mixup boxes but not all used
            
            # makesure the self.boxes_index has enough boxes
            if len(self.boxes_index) < self.batch_size:
                # print("A repeat for boxes memory!")
                self.boxes_index = list(range(len(self.BoxRehearsal_path)))
                
            mixup_count = 0
            for i in range(num_mixup):
                c_img, c_gt, b_id = self._sample_per_bbox_from_boxrehearsal(i, img_shape)
            
                c_img = np.asarray(c_img)
                _c_gt = c_gt.copy()

                # assign a random location
                pos_x = random.randint(0, int(img_shape[1] * 0.6))
                pos_y = random.randint(0, int(img_shape[0] * 0.4))
                new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]

                restart = True
                overlap = False
                max_iter = 0
                # compute the overlap with each gts in image
                while restart:
                    for g in gts:      
                        _, overlap = self.compute_overlap(g, new_gt)
                        if max_iter >= 20:
                            # if iteration > 20, delete current choosed sample
                            restart = False
                        elif max_iter < 10 and overlap:
                            pos_x = random.randint(0, int(img_shape[1] * 0.6))
                            pos_y = random.randint(0, int(img_shape[0] * 0.4))
                            new_gt = [c_gt[0][0] + pos_x, c_gt[0][1] + pos_y, c_gt[0][2] + pos_x, c_gt[0][3] + pos_y]
                            max_iter += 1
                            restart = True
                            break
                        elif 20 > max_iter >= 10 and overlap:
                            # if overlap is True, then change the position at right bottom
                            pos_x = random.randint(int(img_shape[1] * 0.4), img_shape[1])
                            pos_y = random.randint(int(img_shape[0] * 0.6), img_shape[0])
                            new_gt = [pos_x-(c_gt[0][2]-c_gt[0][0]), pos_y-(c_gt[0][3]-c_gt[0][1]), pos_x, pos_y]
                            max_iter += 1
                            restart = True
                            break
                        else:
                            restart = False
                            # print("!!!!{2} the g {0} and new_gt is: {1}".format(g, new_gt, overlap))

                if max_iter < 20:
                    a, b, c, d = 0, 0, 0, 0
                    if new_gt[3] >= img_shape[0]:
                        # at bottom right new gt_y is or not bigger
                        a = new_gt[3] - img_shape[0]
                        new_gt[3] = img_shape[0]
                    if new_gt[2] >= img_shape[1]:
                        # at bottom right new gt_x is or not bigger
                        b = new_gt[2] - img_shape[1]
                        new_gt[2] = img_shape[1]
                    if new_gt[0] < 0:
                        # at top left new gt_x is or not bigger
                        c = -new_gt[0]
                        new_gt[0] = 0
                    if new_gt[1] < 0:
                        # at top left new gt_y is or not bigger
                        d = -new_gt[1]
                        new_gt[1] = 0

                    # Use the formula by the paper to weight each image
                    img1 = Lambda*image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]]
                    c_img = (1-Lambda)*c_img
                    
                    # Combine the images
                    if a == 0 and b == 0:
                        if c == 0 and d == 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:, :]
                        elif c != 0 and d == 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:, c:]
                        elif c == 0 and d != 0:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[d:, :]
                        else:
                            image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[d:, c:]

                    elif a == 0 and b != 0:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:, :-b]
                    elif a != 0 and b == 0:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:-a, :]
                    else:
                        image[new_gt[1]:new_gt[3], new_gt[0]:new_gt[2]] = img1 + c_img[:-a, :-b]

                    _c_gt[0][:-1] = new_gt
                    if gts.shape[0] == 0:
                        gts = _c_gt
                    else:
                        gts = np.insert(gts, 0, values=_c_gt, axis=0)
                    
                    # delete the mixed boxes
                    if b_id in self.boxes_index:
                        self.boxes_index.remove(b_id)

                mixup_count += 1
                if mixup_count>=2:
                    break
        
        Current_image = Image.fromarray(np.uint8(image))
        Current_target = BoxList(gts[:, :4], (img_shape[1], img_shape[0]))
        Current_target.add_field("labels", torch.tensor(gts[:, 4]))
        
        return Current_image, Current_target

    def _start_boxes_mosaic(self, s_imgs=[], targets=[], num_boxes=4):
        """ Start mosaic boxes. 
            A composite image is formed by combining four box images into a single mosaic image
        """
        
        gt4 = [] # the final groundtruth space
        if len(targets)>=1:
            # print(len(s_imgs))
            id = [-1 for i in range(len(s_imgs))] # for the new image
        else:
            id = []
        
        scale = int(np.mean(s_imgs.size)) # keep the same size with current image
        s_w = scale
        s_h = scale
        
        ### FOR without NEW IMAGE
        # The scaling factor mu randomly sampled from the range of [0.4, 0.6]
        yc = int(random.uniform(s_h*0.4, s_h*0.6)) # set the mosaic center position
        xc = int(random.uniform(s_w*0.4, s_w*0.6))
        
        ### FOR ONE NEW IMAGE
        # yc = int(random.uniform(s_h*0.3, s_h*0.4)) # set the mosaic center position
        # xc = int(random.uniform(s_w*0.3, s_w*0.4))
        
        ### preparing the enough box memory for mosaic ###
        if len(self.boxes_index) < self.batch_size:
            # print("A repeat for boxes memory!")
            self.boxes_index = list(range(len(self.BoxRehearsal_path)))
            
        imgs = [] 
        for i in range(num_boxes):
            # put the new images and box memory together
            img, target, b_id = self._sample_per_bbox_from_boxrehearsal(i, s_imgs.size)
            imgs.append(img)
            targets.append(target)
            id.append(b_id)
        
        #### Begin to mosaic ####
        for i, (img, target, b_id) in enumerate(zip(imgs, targets, id)):
            (w, h) = img.size
            if i%4==0: # top right
                xc_ = xc+self.bg_size
                yc_ = yc-self.bg_size
                img4 = np.full((s_h, s_w, 3), 114., dtype=np.float32)
                x1a, y1a, x2a, y2a = xc_, max(yc_-h, 0), min(xc_+w, s_w), yc_
                x1b, y1b, x2b, y2b = 0, h-(y2a - y1a), min(w, x2a - x1a), h # should corresponding to top left
            elif i%4==1: # bottom left
                xc_ = xc-self.bg_size
                yc_ = yc+self.bg_size
                x1a, y1a, x2a, y2a = max(xc_ - w, 0), yc_, xc_, min(s_h, yc_ + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc_, w), min(y2a - y1a, h)
            elif i%4==2: # bottom right
                xc_ = xc+self.bg_size
                yc_ = yc+self.bg_size
                x1a, y1a, x2a, y2a = xc_, yc_, min(xc_ + w, s_w), min(s_h, yc_+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            elif i%4==3: # top left
                xc_ = xc-self.bg_size
                yc_ = yc-self.bg_size
                x1a, y1a, x2a, y2a = max(xc_- w, 0), max(yc_ - h, 0), xc_, yc_
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h-(y2a - y1a), w, h

            img4[y1a:y2a, x1a:x2a] = np.asarray(img)[y1b:y2b,x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            gts = []
            if not isinstance(target, np.ndarray):
                bbox_list = target.bbox.tolist()
                label_list = target.extra_fields["labels"].tolist()
                for i in range(len(bbox_list)):
                    gts.append(bbox_list[i] + [label_list[i]])
            else:
                gts = target

            gts = np.array(gts)
            if len(gts) > 0:
                gts[:, 0] = gts[:, 0] + padw
                gts[:, 1] = gts[:, 1] + padh
                gts[:, 2] = gts[:, 2] + padw
                gts[:, 3] = gts[:, 3] + padh
            gt4.append(gts)

            # delete the mosaiced boxes
            if b_id in self.boxes_index:
                self.boxes_index.remove(b_id)
                
        # Concat/clip gts
        if len(gt4):
            gt4 = np.concatenate(gt4, 0)
            np.clip(gt4[:, 0], 0, s_w, out=gt4[:, 0])
            np.clip(gt4[:, 2], 0, s_w, out=gt4[:, 2])
            np.clip(gt4[:, 1], 0, s_h, out=gt4[:, 1])
            np.clip(gt4[:, 3], 0, s_h, out=gt4[:, 3])

        # Delete too small objects (check again)
        del_index = []
        for col in range(gt4.shape[0]):
            if (gt4[col][2]-gt4[col][0]) <= 2.0 or (gt4[col][3]-gt4[col][1]) <= 2.0:
                del_index.append(col)
        gt4 = np.delete(gt4, del_index, axis=0)

        # == transfer for input == #
        Current_image = Image.fromarray(np.uint8(img4))
        Current_target = BoxList(gt4[:, :4], (s_w, s_h))
        Current_target.add_field("labels", torch.tensor(gt4[:, 4]))

        # Visualize
        # from PIL import ImageDraw
        # a = ImageDraw.ImageDraw(Current_image)
        # for g in range(gt4.shape[0]):
        #     gt_ = gt4[g, :4]
        #     a.rectangle(((gt_[0], gt_[1]), (gt_[2], gt_[3])), fill=None, outline='blue', width=2)
        # Current_image.save('output/mosaic_mixup_boxes/{}.jpg'.format(b_id))

        return Current_image, Current_target

    def _get_index(lst, item):
        return [index for (index, value) in enumerate(lst) if value==item]
    
    def transform_current_data_with_ABR(self, img=None, target=None):
        """ begin to mosaic or mixup the box images into current image """
        
        # set the ratio for replay
        # MIX,MOS,NEW=1:1:2
        is_mosaic = False
        is_mixup = False
        if random.randint(0, 1)==0:
            if random.randint(0, 1)==0:
                is_mixup = True
            else:
                is_mosaic=True
        
        (current_image, current_targets) = (img, target)

        if is_mosaic:
            current_image, current_targets = self._start_boxes_mosaic(current_image, [], num_boxes=4)
        elif is_mixup:
            current_image, current_targets = self._start_mixup(current_image, current_targets)

        # if is_mosaic or is_mixup:
        #     from PIL import ImageDraw
        #     from PIL import ImageFont
        #     a = ImageDraw.ImageDraw(current_image)
        #     # ttf = ImageFont.truetype("/data/users/yliu/workspace/MMA/maskrcnn_benchmark/data/datasets/arial.ttf", 15)
        #     for g in range(current_targets.__len__()):
        #         gt_ = current_targets.bbox.tolist()[g]
        #         l_ = int(current_targets.extra_fields['labels'].tolist()[g])
        #         label_ = self.map_class_id_to_class_name(l_)
        #         if label_ in self.old_classes:
        #             a.rectangle(((gt_[0], gt_[1]), (gt_[2], gt_[3])), fill=None, outline=(0,176,240), width=2)
        #             # a.text((gt_[0]+5, gt_[1]+6), str(label_), font=ttf, fill=(0,0,255))
        #         elif label_ in self.new_classes:
        #             a.rectangle(((gt_[0], gt_[1]), (gt_[2], gt_[3])), fill=None, outline=(237,125,49), width=2)
        #             # a.text((gt_[0]+5, gt_[1]+6), str(label_), font=ttf, fill=(255, 0, 0))
        #     current_image.save('output/box_images/10-10/current_image_{}.jpg'.format(img_id))

        return current_image, current_targets
    
    def get_groundtruth_ABR(self, index):
        """ preparing the groundtruth for current index image """
        img_id = self.final_ids[index]

        anno = ET.parse(self._annopath % img_id).getroot()
        # anno = self._preprocess_annotation(anno)
        anno = self._preprocess_annotation_with_mem(anno, index)
        # print("The index {0}: the anno label is {1}".format(index, anno["labels"]))

        height, width = anno["im_info"]
        self._img_height = height
        self._img_width = width
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation_with_mem(self, target, index):
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

            # only saving the new groundtruth in training, but for testing phase all GTs are saved.
            if exclude_class_flag:
                pass
            elif self.is_sample and old_class_flag:
                pass
            elif self.is_train and old_class_flag:
                pass
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

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset_ABR.CLASSES[class_id]

    def compute_overlap(self, a, b):
        """ compute the overlap of input a and b;
            input: a and b are the box
            output(bool): the overlap > 0.2 or not
        """
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

        iw = np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]) + 1
        ih = np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]) + 1

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        aa = (a[2] - a[0] + 1)*(a[3] - a[1]+1)
        ba = area

        intersection = iw*ih

        # this parameter can be changes for different datasets
        if intersection/aa > 0.3 or intersection/ba > 0.3:
            return intersection/ba, True
        else:
            return intersection/ba, False


if __name__ == '__main__':
    data_dir = "/dataroot/data/voc07/VOCdevkit/VOC2007"
    split = "trainval"  # train, val, test
    use_difficult = False
    transforms = None
    is_train = True

    NAME_OLD_CLASSES=["aeroplane", "bicycle", "bird","boat", "bottle", "bus", "car", "cat", "chair", "cow",
                       "diningtable", "dog", "horse", "motorbike", "person"]
    NAME_NEW_CLASSES=["pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    mem_buff = 2000
    dataset = PascalVOCDataset_ABR(data_dir, split, use_difficult, transforms, is_train=is_train, 
                                old_classes=NAME_OLD_CLASSES, new_classes=NAME_NEW_CLASSES, mem_buff=mem_buff)
