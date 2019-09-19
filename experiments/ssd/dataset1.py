import torch
import torch.nn
from torch.utils.data import Dataset

import cv2
import pickle
import numpy as np
import random

def get_crop_data_augment(image, boxes, labels, img_size):
    height, width, _ = image.shape
    if len(boxes) == 0:
        return image, boxes, labels
    min_range = 0
    boxes[:, 0::2] *= width
    boxes[:, 1::2] *= height

    def get_xd_yd_from_anno(w, h, bbox, labels, min_range):
        assert len(bbox) > 0
        assert len(labels) > 0
        bbox_tmp = bbox.copy()
        selected_index = np.random.choice(range(len(labels)))
        s_bbox = bbox_tmp[selected_index]
        #s_bbox[0::2] *= w
        #s_bbox[1::2] *= h

        min_x = int(max(0, s_bbox[0] + max((s_bbox[2] - s_bbox[0]) / 2, min_range) - img_size))
        min_y = int(max(0, s_bbox[1] + max((s_bbox[3] - s_bbox[1]) / 2, min_range) - img_size))
        max_x = int(min(w - img_size - 1, min(s_bbox[2] - min_range, s_bbox[0])))
        max_y = int(min(h - img_size - 1, min(s_bbox[3] - min_range, s_bbox[1])))

        xd = random.randint(min_x, max_x)
        yd = random.randint(min_y, max_y)

        bbox_tmp[:, 0] -= xd
        bbox_tmp[:, 1] -= yd
        bbox_tmp[:, 2] -= xd
        bbox_tmp[:, 3] -= yd
        return int(xd), int(yd), bbox_tmp, labels

    while True:
        xd, yd, boxes_t, labels_t = get_xd_yd_from_anno(width, height, boxes, labels, min_range)
        for _ in range(50):
            image_t = image[yd: (img_size + yd), xd: (img_size + xd)]
            xrng = [0, img_size - 1]
            yrng = [0, img_size - 1]

            x1 = boxes_t[:, 0]
            y1 = boxes_t[:, 1]
            x2 = boxes_t[:, 2]
            y2 = boxes_t[:, 3]
            keep = np.where((x1 < (xrng[1] - min_range)) & (x2 > (xrng[0] + min_range)) & \
                            (y1 < (yrng[1] - min_range)) & (y2 > (yrng[0] + min_range)))[0]
            boxes_t = boxes_t[keep, :]
            labels_t = labels_t[keep]

            if len(boxes_t) == 0:
                continue

            x1 = x1[keep]
            x2 = x2[keep]
            y1 = y1[keep]
            y2 = y2[keep]

            x1 = np.maximum(xrng[0], x1)
            y1 = np.maximum(yrng[0], y1)
            x2 = np.maximum(xrng[1], x2)
            y2 = np.maximum(yrng[1], y2)

            boxes_t[:, 0] = x1
            boxes_t[:, 1] = y1
            boxes_t[:, 2] = x2
            boxes_t[:, 3] = y2
            return image_t, boxes_t, labels_t

def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha +beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp
    image = image.copy()
    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))
    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp
    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes
    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1, 4)
        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale * scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)
        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, : 2] += (left, top)
        boxes_t[:, 2: ] += (left, top)

        expand_image = np.empty((h, w, depth), dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top: top+height, left: left+width] = image
        image = expand_image

        return image, boxes_t

def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

class dataset(Dataset):
    def __init__(self, meta_file, img_size, mode='xywh'):
        self.meta_file = meta_file
        self.img_size = img_size
        self.mode = mode
        with open(meta_file, 'rb') as f:
            self.files = pickle.load(f)
        self.imgs = []
        for key in self.files.keys():
            self.imgs.append(key)

        self.means = (104, 117, 123)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        meta = self.files[img_path]

        img = cv2.imread(img_path)
        #img = cv2.resize(img, (0, 0), fx=2, fy=2)
        height, width, _ = img.shape
        img, targets = self.preproc(img, meta)
        return img, targets

    def preproc_for_test(self, image, insize, mean):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, (insize, insize),interpolation=interp_method)
        image = image.astype(np.float32)
        image -= mean
        return image.transpose(2, 0, 1)

    def preproc(self, img, meta):
        """
        if img is None:
            img = 
            return 
        if len(meta) == 0:
            return  
        """
        targets = np.array(meta)
        labels = targets[:, 0].copy()
        if targets.shape[1] == 6:
            boxes = targets[:, 2: ]
        elif targets.shape[1] == 5:
            boxes = targets[:, 1: ]

        image_o = img.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = boxes.copy()
        if self.mode == 'xywh':
            boxes_o[:, 0: 2] = boxes[:, 0: 2] - boxes[:, 2: ] / 2
            boxes_o[:, 2: ]  = boxes[:, 0: 2] + boxes[:, 2: ] / 2
        elif self.mode == 'x1y1x2y2':
            boxes_o = boxes.copy()
        labels_o = np.expand_dims(labels.copy(), 1)
        targets_o = np.hstack((boxes_o, labels_o))

        image_t, boxes, labels = get_crop_data_augment(image_o, boxes_o, labels_o, img_size=self.img_size)

        height, width, _ = image_t.shape
        image_t = self.preproc_for_test(image_t, self.img_size, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t) == 0:
            image = preproc_for_test(image_o, self.resize, self.means)
            return torch.from_numpy(image),targets_o

        #labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))
        return torch.from_numpy(image_t), targets_t

    def __len__(self):
        return len(self.files)

def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
