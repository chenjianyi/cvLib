import sys
sys.path.append('../../autoCV')

import torch
import torch.nn as nn

from modules.net import StackHourglass
from heads import CornernetHead

class Model(nn.Module):
    def __init__(self, nstacks, n, dims, modules, num_classes):
        super(Model, self).__init__()
        self.Model = StackHourglass(n, nstacks, dims, modules)
        self.Head = CornernetHead(nstacks, 256, num_classes)

    def forward(self, x, targets=None):
         middle_attrs, outs = self.Model(x)
         if self.training:
             loss, focal_loss, pull_loss, push_loss, regr_loss = self.Head(outs, targets)
             return loss, focal_loss, pull_loss, push_loss, regr_loss
         else:
             tl_heat, br_heat, embedding1, embedding2, tl_regr, br_regr = self.Head(outs)
             return tl_heat, br_heat, embedding1, embedding2, tl_regr, br_regr
