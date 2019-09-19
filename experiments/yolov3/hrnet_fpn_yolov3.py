import sys
sys.path.append('../../autoCV')

import torch
import torch.nn as nn

from modules.net import HighResolutionNet
from heads import YOLOv3Head
from modules.module import FPNMobileNetV3, FPNYoloV3
from basic import convolution2d, residual

from collections import defaultdict

class Pre(nn.Module):
    def __init__(self, bias=False, sn=False, bn=True, act_fun='relu'):
        super(Pre, self).__init__()
        self.conv1 = nn.Sequential(
            convolution2d(3, 32, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            convolution2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
        )
        self.conv2 = nn.Sequential(
            convolution2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            convolution2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        )
        self.conv3 = nn.Sequential(
            convolution2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
            residual(256, 256, kernel_size=3, stride=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)   
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

stage2 = {
    'num_modules': 1, 'num_branches': 2, 'num_blocks': [8, 8], 'num_channels': [128, 256], \
    'block': 'BasicBlock', 'fuse_method': 'sum'
}
stage3 = {
    'num_modules': 1, 'num_branches': 3, 'num_blocks': [4, 4, 4], 'num_channels': [128, 256, 512], \
    'block': 'BasicBlock', 'fuse_method': 'sum'
}
stages = [stage2, stage3]

class Hrnet_FPN_YoloV3(nn.Module):
    def __init__(self, num_classes, anchors, num_levels, img_dim, g_dims, bias=False, sn=True, bn=True, act_fun='relu'):
        super(Hrnet_FPN_YoloV3, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.pre = Pre(bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.Hrnet = HighResolutionNet(stages, pre=self.pre, pre_channels=256, output='feature', bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.Heads = nn.ModuleList()
        n_features = stages[-1]['num_channels']

        
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
        base_features = self.Hrnet(x)
        
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
