import torch
import torch.nn as nn

from basic import convolution2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False, sn=False, bn=True, act_fun='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = convolution2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv2 = convolution2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, sn=sn, bn=bn, act_fun='none')

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    BN_MOMENTUM = 0.1
    def __init__(self, inplanes, planes, stride=1, downsample=None, bias=False, sn=False, bn=True, act_fun='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = convolution2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv2 = convolution2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv3 = convolution2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun='none')

        self.downsample = downsample
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = convolution2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, padding=0, bias=bias, sn=sn, bn=bn, act_fun='none')
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
