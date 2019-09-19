import sys
sys.path.append('../..')
print(sys.path)

import torch
import torch.nn as nn

from modules.net import Darknet53
from heads import YOLOv3Head
from modules.module import FPNYoloV3

from collections import defaultdict

class Darknet53_FPN_YoloV3(nn.Module):
    def __init__(self, num_classes, anchors, num_levels, img_dim, g_dims, bias=True, sn=True, bn=True, act_fun='leakyrelu'):
        super(Darknet53_FPN_YoloV3, self).__init__()
        n_features = [256, 512, 1024]
        self.Darknet53 = Darknet53(output='feature', bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.FPN = FPNYoloV3(n_features, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.Heads = nn.ModuleList()
        for i in range(len(anchors)):
            i_anchors = anchors[i]
            n_feature = n_features[i]
            g_dim = g_dims[i]
            self.Heads.append(YOLOv3Head(i_anchors, num_classes, img_dim, g_dim, n_feature, lambda_coord=0.2))

    def forward(self, x, targets=None):
        base_features = self.Darknet53(x)
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
