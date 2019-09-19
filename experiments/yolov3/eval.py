import os
import sys
import time
import argparse
import pickle
import numpy as np

sys.path.append('..')

import torch
from torch.autograd import Variable
from torchvision import transforms

#from models import Darknet
from darknet53_fpn_yolov3 import Darknet53_FPN_YoloV3
from mobilenetv3_fpn_yolov3 import MobileNetV3_FPN_YoloV3
from m2det_fpn_yolov3 import M2det_FPN_YoloV3
from hrnet_fpn_yolov3 import Hrnet_FPN_YoloV3
from dataset import ListDataset
from _utils import non_max_suppression
from _utils import bbox_iou, compute_ap

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', default=16, type=int, help='size of each mini-batch')
parse.add_argument('--config_path', default='../config/yolov3.cfg', type=str, help='path to model config file')
parse.add_argument('--meta_file', default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/VOC2012_test.pkl', type=str, help='path to image meta file')
parse.add_argument('--label_file', default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/VOC_label_file.txt', type=str, help='path to class name file')
parse.add_argument('--weights_path', default='/data/chenjianyi/records/cvLib/hrnet+fpn+yolov3/2019081613_55.weights', type=str, help='path to weights file')
parse.add_argument('--backbone', type=str, default='hrnet')

parse.add_argument('--iou_thres', default=0.3, type=float, help='iou threshold between predicted boxes and ground-truth boxes')
parse.add_argument('--conf_thres', default=0.3, type=float, help='object cofidence threshold')
parse.add_argument('--nms_thres', default=0.45, type=float, help='iou threshold for non-maximum suppression')

parse.add_argument('--img_size', default=416, type=float, help='size of each image dimension')
parse.add_argument('--use_cuda', default=True, type=bool, help='whether to use cuda if available')

opt = parse.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

with open(opt.label_file, 'r') as f:
    label_list = f.readlines()
label_list = [item.strip() for item in label_list]
label2id_dict = {k: v for v, k in enumerate(label_list)}
id2label_dict = {k: v for k, v in enumerate(label_list)}

num_classes = len(label_list)
num_levels = 8
anchors1 = [[10,13], [16,30], [33,23]]
anchors2 = [[30,61], [62,45], [59,119]]
anchors3 = [[116,90], [156,198], [373,326]]
anchors = [anchors1, anchors2, anchors3]
img_dim = 416
g_dims = [52, 26, 13]

print('Initiate model...')
if opt.backbone == 'darknet53':
    Model = Darknet53_FPN_YoloV3
elif opt.backbone == 'mobilenetv3':
    Model = MobileNetV3_FPN_YoloV3
elif opt.backbone == 'm2det':
    Model = M2det_FPN_YoloV3
elif opt.backbone == 'hrnet':
    Model = Hrnet_FPN_YoloV3
model = Model(num_classes, anchors, num_levels, img_dim, g_dims)
state_dict = torch.load(opt.weights_path)
model.load_state_dict(state_dict)
if cuda:
    model = model.cuda()
model.eval()


print('Load class name...')
with open(opt.label_file, 'r') as f:
    names = f.read().split('\n')[:-1]
dict_labelid_labelname = {k:v for k, v in enumerate(names)}

print(dict_labelid_labelname)

print('Preparing dataset...')
def path_zhoutong(s):
    return s
def get_dataloader(test_path, transform_path):
    dataset = ListDataset(test_path, transform_path, img_size=opt.img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=20)
    return dataset, dataloader

def evaluation(test_path, tranform_path):
    dataset, dataloader = get_dataloader(test_path, tranform_path)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    outputs = []
    targets = None
    APs = []
    t1 = time.time()
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        t1 = time.time()
        imgs = Variable(imgs.type(Tensor))
        targets = targets.type(Tensor)

        with torch.no_grad():
            output = model(imgs)
            output = non_max_suppression(output, 40, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
        t2 = time.time()
        print('consuming:', (t2 - t1))
    
        # Compute average precision for each sample
        for sample_i in range(targets.size(0)):
            correct = []

            # Get labels for sample where width is not zero (dummies)
            annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
            # Extract detections
            detections = output[sample_i]

            if detections is None:
                # If there are no detections but there are annotations mask as zero AP
                if annotations.size(0) != 0:
                    APs.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu()
            annotations = annotations.cpu()
            detections = detections[np.argsort(-detections[:, 4])]

            # If no annotations add number of detections as incorrect
            if annotations.size(0) == 0:
                correct.extend([0 for _ in range(len(detections))])
            else:
                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
                target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
                target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
                target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
                target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
                target_boxes *= opt.img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:

                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > opt.iou_thres:
                        if obj_pred == annotations[best_i, 0] and best_i not in detected:
                            correct.append(1)
                            detected.append(best_i)
                    else:
                        correct.append(0)


            # Extract true and false positives
            true_positives  = np.array(correct)
            false_positives = 1 - true_positives

            # Compute cumulative false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # Compute recall and precision at all ranks
            recall    = true_positives / annotations.size(0) if annotations.size(0) else true_positives
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # Compute average precision
            AP = compute_ap(recall, precision)
            #print(recall, precision, AP)
            APs.append(AP)

    print ("Mean Average Precision: %.4f" % np.mean(APs))

print('Starting evaluation...')
evaluation(opt.meta_file, path_zhoutong)
print(opt)
