import sys
sys.path.append('../..')

import torch
import torch.nn as nn

from modules.net import MobileNetV3_Large
from heads import YOLOv3Head
from modules.module import FPNMobileNetV3, FPNYoloV3
from basic import convolution2d

from collections import defaultdict

class MobileNetV3_FPN_YoloV3(nn.Module):
    def __init__(self, num_classes, anchors, num_levels, img_dim, g_dims, bias=True, sn=False, bn=True, act_fun='leakyrelu'):
        super(MobileNetV3_FPN_YoloV3, self).__init__()
        #n_features = [40, 160, 160]
        #self.FPN = FPNMobileNetV3(n_features, bias=bias, sn=sn, bn=bn)
        n_features = [256, 512, 1024]
        self.FPN = FPNYoloV3(n_features, bias=bias, sn=sn, bn=bn)
        self.MobileNetV3 = MobileNetV3_Large(output='feature', bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.Mids = nn.ModuleList([
            convolution2d(40, 256, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            convolution2d(160, 512, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            convolution2d(160, 1024, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
        ])
        self.Heads = nn.ModuleList()
        for i in range(len(anchors)):
            i_anchors = anchors[i]
            n_feature = n_features[i]
            g_dim = g_dims[i]
            self.Heads.append(YOLOv3Head(i_anchors, num_classes, img_dim, g_dim, n_feature, lambda_coord=0.2))

    def forward(self, x, targets=None):
        base_features = self.MobileNetV3(x)[1: ]
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
