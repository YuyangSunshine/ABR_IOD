#-*-coding:utf-8-*-

from maskrcnn_benchmark.config import \
    cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file

import argparse
import random
import os
import math
import shutil
import numpy as np
import pickle

from PIL import Image, ImageFilter


class Mem(object):
	def __init__(self, cfg, step=0, current_mem_path=None):
		"""
		construct memory and update memory
		under some different strategies
		:param new_classes: the number of current classes
		:param mem_size: memory size
		:param old_classes: previous memory classes
		"""
		
		self.new_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES
		self.old_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES
		self.all_classes = self.old_classes + self.new_classes
		self.cfg = cfg
		self.mem_type = self.cfg.MEM_TYPE
		self.mem_size = self.cfg.MEM_BUFF
		self.STEP = step

		# dataset path
		self.root = "data/VOCdevkit/VOC2007"
		self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
		
		# memory name space
		self.current_mem_name = f"{self.mem_type}_{self.mem_size}"
		self.current_mem_path = current_mem_path
		
		# loading the old memory
		self.exemplar = None # the list for box image path name

		# current step >= 1: need to review the last step information
		if self.STEP ==0:
			self.exemplar = os.listdir(self.current_mem_path)
		elif self.STEP == 1:
			self.first_mem_path = os.path.join(os.path.split(self.cfg.MODEL.SOURCE_WEIGHT)[0], self.current_mem_name)
			self.exemplar = os.listdir(self.first_mem_path)
			first_num_file_classes = len(self.exemplar)
			assert first_num_file_classes >= self.mem_size, 'The selected rehearsals are not satisfied the setting size!'
		elif self.STEP>1:
			self.current_mem_path = os.path.join(f"output/{self.cfg.TASK}/{self.cfg.NAME}", self.current_mem_name)
			self.exemplar = os.listdir(self.current_mem_path)

		print('--PBS REPORT-- Current prototype boxes path is {0}'.format(self.current_mem_path))
		
		self.num_current_classes = len(self.new_classes)
		self.num_bbox_per_cls = math.ceil(self.mem_size / len(self.all_classes))

		self.current_mem_info = []
		self.current_features = []
		self.current_logits = []

	def get_fea_log_classes(self, mem_info):
		num_classes = len(mem_info)
		assert num_classes==self.num_current_classes
		features = [[] for _ in range(num_classes)]
		logits = [[] for _ in range(num_classes)]

		for i in range(num_classes):
			each_cls_len = len(mem_info[i])
			print('-- Class {} have the features is: {}'.format(i, each_cls_len))
			features[i] = [mem_info[i][j]['feature'] for j in range(each_cls_len)]
			logits[i] = [mem_info[i][j]['logits'] for j in range(each_cls_len)]

		self.current_mem_info = mem_info

		return features, logits

	def rnd_sampling(self):
		# random by the info information
		_ind_bbox_per_cls = 0
		for i in range(self.num_current_classes):
			print("In random sampling and current classes is: {}".format(i))
			
			# Shuffle the current class's memory information
			random.shuffle(self.current_mem_info[i])

			# If the number of box images for the current class is less than the desired number (self.num_bbox_per_cls)
            # Fill with some of the same box images
			if len(self.current_mem_info[i])<self.num_bbox_per_cls:
				self.current_mem_info[i].extend(self.current_mem_info[i][:self.num_bbox_per_cls-len(self.current_mem_info[i])])
			
			for j in range(len(self.current_mem_info[i])):
				if _ind_bbox_per_cls<self.num_bbox_per_cls:
					self.creat_and_save_box_image(self.current_mem_info[i][j], _ind_bbox_per_cls)
					_ind_bbox_per_cls += 1
				else:
					break

			_ind_bbox_per_cls = 0

		# Check if the number of saved images in the current directory meets the memory size requirement			
		assert len(os.listdir(self.current_mem_path))>=self.mem_size, 'The selected rehearsals are not satisfied the setting size!'
		
		return os.listdir(self.current_mem_path)

	def mean_feature_sampling(self):
		class_mean = [[] for _ in range(self.num_current_classes)]

		_ind_bbox_per_cls = 0
		for i in range(self.num_current_classes):
			print("In mean sampling and current classes is: {}".format(i))
			# If the number of box images for the current class is less than the desired number (self.num_bbox_per_cls)
            # Fill with some of the same box images
			if len(self.current_mem_info[i])<self.num_bbox_per_cls:
				deficit = self.num_bbox_per_cls - len(self.current_mem_info[i])
				self.current_mem_info[i].extend(self.current_mem_info[i][:deficit])
				self.current_features[i].extend(self.current_features[i][:deficit])
				self.current_logits[i].extend(self.current_logits[i][:deficit])

			boxes_fea = np.array(self.current_features[i])
			# print('--LYY REPORT-- class {} have the features shape is: {}'.format(y, boxes_fea.shape))

			if not class_mean[i]:
				# Calculate the mean feature for the class
				cls_mean = np.mean(boxes_fea, axis=0)
				cls_mean /= np.linalg.norm(cls_mean)
				class_mean[i] = cls_mean
			else:
				cls_mean = class_mean[i]

			phi = boxes_fea
			mu = cls_mean

			phi /= np.linalg.norm(phi)
			# Calculate the distance between the class mean and features
			dist = np.sqrt(np.sum((mu - phi) ** 2, axis=(1,2)))
			
			# Sort and select exemplars based on distance
			memory_index = np.argsort(dist)
			memory_index = memory_index[:self.num_bbox_per_cls]
			self.current_mem_info[i] = [self.current_mem_info[i][ind] for ind in memory_index]
			
			# Create and save box images
			for j in range(len(self.current_mem_info[i])):
				if _ind_bbox_per_cls < self.num_bbox_per_cls:
					self.creat_and_save_box_image(self.current_mem_info[i][j], _ind_bbox_per_cls)
					_ind_bbox_per_cls += 1
				else:
					break

			_ind_bbox_per_cls = 0

		# Check if the number of saved images in the current directory meets the memory size requirement			
		assert len(os.listdir(self.current_mem_path))>=self.mem_size, 'The selected rehearsals are not satisfied the setting size!'
		
		return os.listdir(self.current_mem_path)

	def herding_feature_sampling(self):
		class_mean = [[] for _ in range(self.num_current_classes)]
		current_center = 0

		for i in range(self.num_current_classes):
			# If the number of box images for the current class is less than the desired number (self.num_bbox_per_cls)
            # Fill with some of the same box images
			if len(self.current_mem_info[i])<self.num_bbox_per_cls:
				deficit = self.num_bbox_per_cls - len(self.current_mem_info[i])
				self.current_mem_info[i].extend(self.current_mem_info[i][:deficit])
				self.current_features[i].extend(self.current_features[i][:deficit])
				self.current_logits[i].extend(self.current_logits[i][:deficit])

			boxes_fea = np.array(self.current_features[i])
			(N, W, H) = boxes_fea.shape
			boxes_fea = np.reshape(boxes_fea, (N, W*H))

			if not class_mean[i]:
				# Calculate the mean feature for the class
				cls_mean = np.mean(boxes_fea, axis=0)
				cls_mean /= np.linalg.norm(cls_mean)
				class_mean[i] = cls_mean
			else:
				cls_mean = class_mean[i]
   
			current_center = cls_mean * 0
			selected_indices = []

			for f in range(len(boxes_fea)):
				# Compute distances with the current center
				candidate_centers = current_center * f / (f + 1) + boxes_fea / (f + 1)
				distances = pow(candidate_centers - cls_mean, 2).sum(axis=1)
				distances[selected_indices] = np.inf
				
				# Select the best candidate
				new_index = distances.argmin().tolist()
				selected_indices.append(new_index)
				current_center = candidate_centers[new_index]

			# Sort and select exemplars based on distance
			memory_index = selected_indices[:self.num_bbox_per_cls]
			self.current_mem_info[i] = [self.current_mem_info[i][ind] for ind in memory_index]
			
			# Create and save box images
			for j in range(len(self.current_mem_info[i])):
				if _ind_bbox_per_cls < self.num_bbox_per_cls:
					self.creat_and_save_box_image(self.current_mem_info[i][j], _ind_bbox_per_cls)
					_ind_bbox_per_cls += 1
				else:
					_ind_bbox_per_cls = 0
					break

		# Check if the number of saved images in the current directory meets the memory size requirement			
		assert len(os.listdir(self.current_mem_path))>=self.mem_size, 'The selected rehearsals are not satisfied the setting size!'
		
		return os.listdir(self.current_mem_path)

	def creat_and_save_box_image(self, bbox_info, ind):	
		# Get image path, bounding boxes, and class information from bbox_info	
		im_path = bbox_info['image_path'][0]
		bboxes = bbox_info['box']
		gt_classes = bbox_info['box_class']
		
		# Open and convert the image to RGB
		im = Image.open(self._imgpath % im_path).convert("RGB")
		
		# Crop the image based on the bounding box coordinates
		box_im = im.crop((int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])))

		# Creat the box image path "class_index.jpg"
		box_im_path = "{0}_{1:05d}.jpg".format(gt_classes, ind)

		# Save the box image
		box_im.save(os.path.join(self.current_mem_path,box_im_path))

	def update_memory(self, input_bboxes_info):

		if self.STEP==0 and input_bboxes_info is None:
			return
		elif self.STEP==1:
			for file_name in self.exemplar:
				full_file_name = os.path.join(self.first_mem_path, file_name)
				cls_name, index = os.path.splitext(file_name)[0].split('_')
				if os.path.isfile(full_file_name) and int(index)<=self.num_bbox_per_cls-1:
					shutil.copy(full_file_name, self.current_mem_path)
		elif self.STEP>1:
			for file_name in self.exemplar:
				full_file_name = os.path.join(self.current_mem_path, file_name)
				cls_name, index = os.path.splitext(file_name)[0].split('_')
				if os.path.isfile(full_file_name) and int(index)>self.num_bbox_per_cls-1:
					os.remove(full_file_name)		
			
		print('The old classes have {} box images.'.format(len(os.listdir(self.current_mem_path))))

		self.current_features, self.current_logits = self.get_fea_log_classes(input_bboxes_info)

		# Choose the corrsponding sampling method
		if self.mem_type == 'random':
			self.exemplar = self.rnd_sampling()
		elif self.mem_type == 'mean':
			self.exemplar = self.mean_feature_sampling()
		elif self.mem_type == 'herding':
			self.exemplar = self.herding_feature_sampling()

		print("--PBS REPORT-- The Box Rehearsals are saved in {}".format(self.current_mem_path))
  

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
      parser.add_argument(
		"-t", "--task",
		type=str,
		default="15-5"
	)
      parser.add_argument(
            "-n", "--name",
            default="MMA_ms_15_5",
      		)
      parser.add_argument(
            "-mb", "--memory_buffer",
            default=2000, type=int,
      )
      parser.add_argument(
            "-mt", "--memory_type",
            default="mean", type=str,
      )
      parser.add_argument(
            "-s", "--step",
            default=0, type=int,
	)

	# # assign the gpu
	# # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

      args = parser.parse_args()
	# setting the corresponding GPU
      # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number
      if args.step == 0:
            source_model_config_file = f"configs/OD_cfg/{args.task}/e2e_faster_rcnn_R_50_C4_4x_RB.yaml"
      else:
            source_model_config_file = f"configs/OD_cfg/{args.task}/e2e_faster_rcnn_R_50_C4_4x_RB_Target_model.yaml"
      full_name = f"{args.name}/"  # if args.step > 1 else args.name
      
      cfg_source = cfg.clone()
      cfg_source.merge_from_file(source_model_config_file)
      
      cfg_source.MODEL.WEIGHT = cfg_source.MODEL.SOURCE_WEIGHT
      cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES) + 1
      
      cfg_source.OUTPUT_DIR += args.task + "/" + full_name
      cfg_source.TENSORBOARD_DIR += args.task + "/" + full_name
      cfg_source.TASK = args.task
      cfg_source.STEP = args.step
      cfg_source.NAME = args.name
      cfg_source.MEM_BUFF = args.memory_buffer
      cfg_source.MEM_TYPE = args.memory_type
      cfg_source.freeze()
      
      _imgpath = os.path.join("/data/users/yliu/workspace/MMA/data/voc07/VOCdevkit/VOC2007", "JPEGImages", "%s.jpg")
      
      Rehearsal_Memory = Mem(cfg_source)
      RM = []
      for cls in range(len(Rehearsal_Memory.exemplar['info'])):
            if cls == 0:
                  RM=Rehearsal_Memory.exemplar['info'][cls]
            else:
                  RM.extend(Rehearsal_Memory.exemplar['info'][cls])
                  
      boxes_index = list(range(len(RM)))
      for i in boxes_index:
            bbox_info_ran_class = RM[i]
            
            im_path = bbox_info_ran_class['image_path'][0]
            bboxes = bbox_info_ran_class['box']
            gt_classes = bbox_info_ran_class['box_class']
            
            
            im = Image.open(_imgpath % im_path).convert("RGB")
            im_shape = im.size
            
            box_im = im.crop((int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])))
            
            box_im.save('./output/box_images/{0}/box_org_im_{1}.jpg'.format(args.task, i))
               
      