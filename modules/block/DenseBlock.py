import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from basic import convolution2d

class DenseLayer(nn.Module):
    def __init__(self, inp_planes, mid_planes=192, growth_rate=48, bias=False, sn=False, bn=True, act_fun='relu'):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            convolution2d(inp_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False),
            convolution2d(mid_planes, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False)
        )

    def forward(self, x):
        y = self.conv(x)
        y = torch.cat([x, y], 1)
        return y

class DenseBlock(nn.Module):
    def __init__(self, inp_planes, mid_planes, growth_rate, layer_num, bias=False, sn=False, bn=True, act_fun='relu'):
        super(DenseBlock, self).__init__()
        layers = []
        layers.append(DenseLayer(inp_planes, mid_planes, growth_rate, bias=bias, sn=sn, bn=bn, act_fun=act_fun))
        for layer_idx in range(1, layer_num):
            layers.append(DenseLayer(inp_planes+growth_rate*layer_idx, mid_planes, growth_rate))
        self.dense = nn.Sequential(*layers)
        
    def forward(self, x):
        # input:  bs * inp_planes * h * w
        # output: bs * (inp_planes + growth_rate * layer_num) * h * w
        return self.dense(x)
