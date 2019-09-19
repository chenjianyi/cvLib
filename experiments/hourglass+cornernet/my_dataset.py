import os
import numpy as np
import pickle
import cv2
import math

import torch
from torch.utils.data import Dataset

from _utils import *

config = {
    'rand_flip': False,
    'rand_flip_prob': 0.1,

    'rand_scale': [1],
    'rand_crop': False,
    'rand_color': False,
    'lighting': True,

    'border': 128,
    'gaussian_bump': True,
    'gaussian_iou': 0.3,
    'gaussian_rad': -1,

    'mean': np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32),
    'std': np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32),
    'eig_val': np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32),
    'eig_vec': np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32),

    'input_size': [511, 511],
    'output_size': [128, 128],
}

class Dataset():
    def __init__(self, meta_file, label_file):
        with open(meta_file, 'rb') as f:
            self.files = pickle.load(f)
        self.imgs = []
        for key in self.files.keys():
            self.imgs.append(key)
        with open(label_file, 'r') as f:
            self.label_list = [line.strip() for line in f]
        self.label2id = {k: v for v, k in enumerate(self.label_list)}
        self.id2label = {k: v for k, v in enumerate(self.label_list)}
        self.num_classes = len(self.label_list)
        self.config = config

    def __getitem__(self, index):
        img_path = self.imgs[index]
        meta = np.array(self.files[img_path])

        # allocating memory
        num_classes = len(self.label_list)
        max_tag_len = 128
        tl_heatmaps = np.zeros((num_classes, self.config['output_size'][0], self.config['output_size'][1]), dtype=np.float32)
        br_heatmaps = np.zeros((num_classes, self.config['output_size'][0], self.config['output_size'][1]), dtype=np.float32)
        tl_regrs = np.zeros((max_tag_len, 2), dtype=np.float32)
        br_regrs = np.zeros((max_tag_len, 2), dtype=np.float32)
        tl_tags = np.zeros((max_tag_len, ), dtype=np.int64)
        br_tags = np.zeros((max_tag_len, ), dtype=np.int64)
        tag_masks = np.zeros((max_tag_len, ), dtype=np.uint8)
        tag_len = 0
        
        # reading image
        img = cv2.imread(img_path)
        h, w, c = img.shape
        
        # cropping an image randomly
        if self.config['rand_crop']:
            image, detections = random_crop(img, meta, self.config['rand_scale'], self.config['input_size'], \
                                            border=self.config['border'])
        else:
            image, detections = full_image_crop(img, meta)

        # resize & clip
        image, detections = resize_image(image, detections, self.config['input_size'])
        image = image.astype(np.float32) / 255.
        detections = clip_detections(image, detections)

        width_ratio = self.config['output_size'][1] / self.config['input_size'][1]
        height_ratio = self.config['output_size'][0] / self.config['input_size'][0]

        # flipping an image randomly
        if self.config['rand_flip']:
            if np.random.uniform() < self.config['rand_flip_prob']:
                image[:] = image[:, ::-1, :]
                width = image.shape[1]
                detections[:, [0, 2]] = width - 1 - detections[:, [2, 0]]

        # random color
        if self.config['rand_color']:
            color_jittering(data_rng, image)
            # lighting
            if self.config['lighting']:
                lighting(data_rng, image, 0.1, config['eig_val'], config['eig_vec'])

        # normalize
        normalize(image, config['mean'], config['std'])

        # (h, w, c) -> (c, h, w)
        image = image.transpose((2, 0, 1))

        # **** # Processing detections
        for ind, detection in enumerate(detections):
            category = int(detection[4])
            x_top_left, y_top_left = detection[0], detection[1]
            x_bottom_right, y_bottom_right = detection[2], detection[3]

            fxtl = (x_top_left * width_ratio)
            fytl = (y_top_left * height_ratio)
            fxbr = (x_bottom_right * width_ratio)
            fybr = (y_bottom_right * height_ratio)

            xtl, ytl = int(fxtl), int(fytl)
            xbr, ybr = int(fxbr), int(fybr)

            if self.config['gaussian_bump']:
                width = detection[2] - detection[0]
                height = detection[3] - detection[1]
                width = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if self.config['gaussian_rad'] == -1:
                    radius = gaussian_radius((height, width), self.config['gaussian_iou'])
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad
                draw_gaussian(tl_heatmaps[category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[category], [xbr, ybr], radius)
            else:
                tl_heatmaps[category, ytl, xtl] = 1
                br_heatmaps[category, ybr, xbr] = 1
            
            tag_ind = tag_len
            tl_regrs[tag_ind, :] = [fxtl - xtl, fytl - ytl]
            br_regrs[tag_ind, :] = [fxbr - xbr, fybr - ybr]
            tl_tags[tag_ind] = ytl * self.config['output_size'][1] + xtl
            br_tags[tag_ind] = ybr * self.config['output_size'][1] + xbr
            tag_len += 1

        tag_masks[: tag_len]  = 1

        image = torch.from_numpy(image)
        tl_heatmaps = torch.from_numpy(tl_heatmaps)
        br_heatmaps = torch.from_numpy(br_heatmaps)
        tl_regrs = torch.from_numpy(tl_regrs)
        br_regrs = torch.from_numpy(br_regrs)
        tl_tags = torch.from_numpy(tl_tags)
        br_tags = torch.from_numpy(br_tags)
        tag_masks = torch.from_numpy(tag_masks)

        output = {\
            'xs': image, \
            'ys': (tl_heatmaps, br_heatmaps, tl_tags, br_tags, tl_regrs, br_regrs, tag_masks) \
        }

        return output

    def __len__(self):
        return len(self.files)
