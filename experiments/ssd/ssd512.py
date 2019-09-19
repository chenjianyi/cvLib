import sys
sys.path.append('../..')

import torch
import torch.nn as nn

from modules.net import VGG_SSD512
from heads import SSDHead

from collections import defaultdict

priors_setting = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [25.6, 48.0, 105.6, 163.2, 220.8, 278.4, 278.4],
    'max_sizes': [48.0, 105.6, 163.2, 220.8, 278.4, 320.0, 336.0],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True
}

loss_setting = {
    'overlap_thresh': 0.2,
    'negpos_ratio': 3
}

class SSD512(nn.Module):
    def __init__(self, num_classes, num_scales, num_anchors, num_levels, sn=True):
        super(SSD512, self).__init__()
        self.VGG_SSD512 = VGG_SSD512(output='feature', sn=sn)
        loss_setting['bkg_label'] = num_classes
        num_classes += 1
        loss_setting['num_classes'] = num_classes
        self.num_classes = num_classes
        inplanes = [512, 1024, 512, 256, 256, 256, 256]
        #num_anchors = num_scales
        self.Head = SSDHead(inplanes, num_anchors, num_scales, num_classes, loss_setting, priors_setting)

    def forward(self, x, targets=None):
        base_features = self.VGG_SSD512(x)
        if self.training:
            loss, loss_l, loss_c = self.Head(base_features, targets)
            return loss, loss_l, loss_c
        else:
            outputs = self.Head(base_features)
            return outputs
