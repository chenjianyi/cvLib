import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from basic import residual, convolution2d
from module import HourglassModule

class StackHourglass(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, pre=None, cnv_dim=256, layer=residual, **kwagrs
    ):
        super(StackHourglass, self).__init__()
        self.nstack = nstack  # nstack = 2

        # dims: dims = [256, 256, 384, 384, 384, 512]
        # modules: modules = [2, 2, 2, 2, 2, 4]
        # n = 5
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution2d(3, 128, 7, stride=2, padding=3),
            residual(128, 256, 3, stride=2)
        ) if pre is None else pre

        HourglassModule(n, dims, modules, layer=layer)
        self.kps = nn.ModuleList([
            HourglassModule(
                n, dims, modules, layer=layer
            ) for _ in range(nstack)
        ])

        self.cnvs = nn.ModuleList([
            convolution2d(dims[0], cnv_dim, kernel_size=3, stide=1, padding=1, bn=True) for _ in range(nstack)
        ])

        self.inters1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack-1)
        ])
        self.inters2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim),
            ) for _ in range(nstack-1)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.inters3 = nn.ModuleList([
            residual(curr_dim, curr_dim, kernel_size=3, stride=1) for _ in range(nstack-1)
        ])

    def forward(self, x):
        inter = self.pre(x)
        layers = zip(self.kps, self.cnvs)
        middle_attrs = []
        outs = []
        for ind, layer in enumerate(layers):
            kp_, cnv_ = layer[0: 2]
            kp, attrs = kp_(inter)
            cnv = cnv_(kp)
            middle_attrs.append(attrs)
            outs.append(cnv)

            if ind < self.nstack - 1:
                inter = self.inters1[ind](inter) + self.inters2[ind](cnv)  # bs * c * h * w
                inter = self.relu(inter)
                inter = self.inters3[ind](inter)
        return middle_attrs, outs

def test():
    x = torch.Tensor(1, 3, 512, 512).uniform_()
    n = 5
    dims = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    n = 3
    dims = [256, 256, 384, 384]
    modules = [2, 2, 2, 4]
    nstacks = 2
    net = StackHourglass(n, nstacks, dims, modules)
    middle_attrs, outs = net(x)
    for out in middle_attrs:
        print([item.size() for item in out])

if __name__ == '__main__':
    test()
