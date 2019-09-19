import sys
sys.path.append('../../autoCV')

import torch
import torch.nn as nn

from modules.net import M2Det
from heads import YOLOv3Head
from modules.module import FPNMobileNetV3, FPNYoloV3
from basic import convolution2d

from collections import defaultdict

class M2det_FPN_YoloV3(nn.Module):
    def __init__(self, num_classes, anchors, num_levels, img_dim, g_dims, bias=False, sn=True, bn=True, act_fun='relu'):
        super(M2det_FPN_YoloV3, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.M2Det = M2Det(num_scales=len(anchors), num_levels=num_levels, planes=256)
        self.Heads = nn.ModuleList()
        n_features = [256 * num_levels] * 3

        self.Mids = nn.ModuleList([
            convolution2d(n_features[0], 256, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            convolution2d(n_features[1], 512, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            convolution2d(n_features[2], 1024, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
        ])
        n_features = [256, 512, 1024]
        self.FPN = FPNYoloV3(n_features, bias=bias, sn=sn, bn=bn)

        for i, i_anchors in enumerate(anchors):
            g_dim = g_dims[i]
            self.Heads.append(YOLOv3Head(i_anchors, num_classes, img_dim, g_dim, n_features[i], lambda_coord=0.2))

    def forward(self, x, targets=None):
        base_features = self.M2Det(x)

        for i in range(len(base_features)):
            base_features[i] = self.Mids[i](base_features[i])
        base_features = self.FPN(base_features)

        if self.training:
            self.losses = defaultdict(float)
            self.loss_names = ['xy', 'wh', 'conf', 'cls', 'recall']
            output = []
            for i, Head in enumerate(self.Heads):
                loss, *losses = Head(base_features[i], targets[i])
                for name, loss_ in zip(self.loss_names, losses):
                    self.losses[name] += loss_
                    output.append(loss)
            self.losses['recall'] /= 3

        else:
            output = []
            for i, Head in enumerate(self.Heads):
                x = Head(base_features[i])
                output.append(x)
    
        return sum(output) if self.training else torch.cat(output, 1)
