import sys
sys.path.append('../..')

import torch
import torch.nn as nn

from modules.net import M2Det
from heads import SSDHead

from collections import defaultdict

priors_setting = {
    'feature_maps': [40, 20, 10, 5, 3, 2],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 107, 320],
    'min_sizes': [25.6, 48.0, 105.6, 163.2, 220.8, 278.4],
    'max_sizes': [48.0, 105.6, 163.2, 220.8, 278.4, 336.0],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True
}

loss_setting = {
    'overlap_thresh': 0.2,
    'negpos_ratio': 3
}

class M2det_SSD(nn.Module):
    def __init__(self, num_classes, num_scales, num_anchors, num_levels, sn=True):
        super(M2det_SSD, self).__init__()
        self.M2Det = M2Det(num_scales=num_scales, num_levels=num_levels, planes=256)
        loss_setting['bkg_label'] = num_classes
        num_classes += 1
        loss_setting['num_classes'] = num_classes
        self.num_classes = num_classes
        inplanes = 256 * num_levels
        num_anchors = num_scales
        self.Head = SSDHead(inplanes, num_anchors, num_scales, num_classes, loss_setting, priors_setting)

    def forward(self, x, targets=None):
        base_features = self.M2Det(x)
        if self.training:
            loss, loss_l, loss_c = self.Head(base_features, targets)
            return loss, loss_l, loss_c
        else:
            outputs = self.Head(base_features)
            return outputs
