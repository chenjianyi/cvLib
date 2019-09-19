import os
import sys
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from models import Model
import numpy as np
import cv2, pickle

from utils2 import weights_init_normal
from utils2 import TargetBuilder, TargetBuilder2
from dataset import ObjDataset2, detection_collate

parser = argparse.ArgumentParser()
parser.add_argument('--meta_file', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/0719/test_seg2_2.pkl', help='the path of train set file')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--weights_path', type=str, default='/data/chenjianyi/records/m2det2/2019072519_600.weights', help='optional. path to weights file')
parser.add_argument('--label_file', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/label_file.txt', help='mapping of label_index to label_name')
parser.add_argument('--img_size', type=int, default=320)

parser.add_argument('--nms_thres', type=float, default=0.4, help='iou threshold for non-maxinum suppression')
parser.add_argument('--gpu', type=str, default='0', help='which device to use for train')
opt = parser.parse_args()
print(opt)

def nms(boxes, confs, bkg_label, nms_thres=0.3):
    output = [None] * len(boxes)
    for i in range(len(boxes)):
        conf = confs[i]
        box = boxes[i]
        class_conf, class_pred = torch.max(conf, 1, keepdim=True)
        keep = class_pred != bkg_label
        detections = torch.cat([box, class_conf.float(), class_pred.float()], 1)
        unique_labels = detections[:, -1].cpu().unique()
        if boxes.is_cuda:
            unique_labels = unique_labels.cuda()
  
        for c in unique_labels:
            if int(c.item()) == int(bkg_label):
                continue
            detections_class = detections[detections[:, -1] == c]
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1: ])
                detections_class = detections_class[1: ][ious < nms_thres]
            max_detections = torch.cat(max_detections).data
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
    return output

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

class Detector():
    def __init__(self, opt):
        self.opt = opt
        with open(opt.meta_file, 'rb') as f:
            self.files = pickle.load(f)
        with open(opt.label_file, 'r') as f:
            label_list = f.readlines()
        label_list = [item.strip() for item in label_list]
        self.num_classes = len(label_list)
        label2id_dict = {k: v for v, k in enumerate(label_list)}
        id2label_dict = {k: v for k, v in enumerate(label_list)}
        self.model = self.reload_model()

    def reload_model(self):
        num_classes = self.num_classes
        num_levels = 6
        num_anchors = 6
        num_scales = 6
        model = Model(num_classes, num_scales, num_anchors, num_levels)
        state_dict = torch.load(self.opt.weights_path)
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()
        return model

    def get_item(self, path):
        img = cv2.imread(img_path)
        origin_img = img.copy()
        h, w, c = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ( (pad1, pad2), (0, 0), (0, 0) ) if h <= w else ( (0, 0), (pad1, pad2), (0, 0)  )
        pad_h = pad1 if h <= w else 0
        pad_w = pad1 if h >  w else 0
        input_img = np.pad(img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = input_img.shape
        input_img = cv2.resize(input_img, (self.opt.img_size, self.opt.img_size))
        input_img = np.transpose(input_img, (2, 0, 1)) #Channels-first
        input_img = torch.from_numpy(input_img).float()
        input_img = torch.unsqueeze(input_img, 0).cuda()

        labels = self.files[path]
        if len(labels) == 0:
            return input_img, None
        labels = np.array(labels)
        if labels[0].shape[0] == 6:
            labels = np.delete(labels, 1, axis=1)  # (label, x, y, w, h)
        origin_labels = labels.copy()
        x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        y2 = h * (labels[:, 2] + labels[:, 4] / 2)
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h
        labels = torch.from_numpy(labels)
        return input_img, labels, origin_img, origin_labels, (pad_w, pad_h)

    def predict(self, path):
        input_img, labels, origin_img, origin_labels, pads = self.get_item(path)
        scale = max(origin_img.shape[0], origin_img.shape[1])
        boxes, confs = self.model(input_img)
        bkg_label = self.num_classes
        boxes = boxes * scale
        #class_conf, class_pred = torch.max(confs, 1, keep_dim=True)
        outputs = nms(boxes, confs, bkg_label=self.num_classes)
        for i in range(len(outputs)):
            if outputs[i] is None:
                continue
            outputs[i][:, 0::2] -= pads[0]
            outputs[i][:, 1::2] -= pads[1]
            output = outputs[i]
            keep = (output[:, 0] < output[:, 2]) & (output[:, 1] < output[:, 3]) & (output[:, 0] >= 0) & (output[:, 1] >= 0)
            outputs[i] = outputs[i][keep]
        self.draw(origin_img, outputs[0], path)
        return outputs

    def draw(self, img, output, img_path):
        img_name = img_path.split('/')[-1]
        if output is not None:
            for x1, y1, x2, y2, conf, cls in output:
                cls = int(cls)
                left_up = (int(x1), int(y1))
                right_down = (int(x2), int(y2))
                centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.rectangle(img, left_up, right_down, (0, 0, 255))
                cv2.putText(img, str(cls), centre, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        save_path = os.path.join('results', img_name)
        cv2.imwrite(save_path, img)

if __name__ == '__main__':
    detector = Detector(opt)
    with open(opt.meta_file, 'rb') as f:
        files = pickle.load(f)
    for img_path in files:
        print(img_path)
        outputs = detector.predict(img_path)
        output = outputs[0]
        
