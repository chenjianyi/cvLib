import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from normalization import SwitchNorm2d

class convolution2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, \
                 groups=1, act_fun='relu', sn=False, bn=False, bias=True, conv_first=True, transpose=False, **kwargs):
        super(convolution2d, self).__init__()
        self.conv_first = conv_first
        if not transpose:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, \
                                  dilation=dilation, groups=groups, bias=bias)
        else:
            output_padding = kwargs.get('output_padding') if kwargs.get('output_padding') else 0
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, \
                                           output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

        eps = kwargs.get('eps') if kwargs.get('eps') else 1e-5
        momentum = kwargs.get('momentum') if kwargs.get('momentum') else 0.1
        affine = kwargs.get('affine') if kwargs.get('affine') else True
        track_running_stats = kwargs.get('track_running_stats') if kwargs.get('track_running_stats') else True
        out_planes = out_planes if conv_first else in_planes
        if sn == False:
            self.bn = nn.BatchNorm2d(out_planes, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats) if bn else nn.Sequential()
        else:
            self.bn = SwitchNorm2d(out_planes, eps=eps, momentum=momentum, using_bn=bn)
        
        if act_fun.lower() == 'relu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.ReLU(inplace)
        elif act_fun.lower() == 'leakyrelu':
            negative_slope = kwargs.get('negative_slope') if kwargs.get('negative_slope') else 0.01
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.LeakyReLU(negative_slope, inplace)
        elif act_fun.lower() == 'relu6':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.ReLU6(inplace)
        elif act_fun.lower() == 'rrelu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            lower = kwargs.get('lower') if kwargs.get('lower') else 0.125
            upper = kwargs.get('upper') if kwargs.get('upper') else 0.3333333333333333
            self.act_f = nn.RReLU(lower=lower, upper=upper, inplace=inplace)
        elif act_fun.lower() == 'selu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            self.act_f = nn.SELU(inplace=inplace)
        elif act_fun.lower() == 'celu':
            inplace = kwargs.get('inpace') if kwargs.get('inpace') else True
            alpha = kwargs.get('alpha') if kwargs.get('alpha') else 1.0
            self.act_f = nn.CELU(alpha=alpha, inplace=inplace)
        elif act_fun.lower() == 'sigmoid':
            self.act_f = nn.Sigmoid()
        elif act_fun.lower() == 'softplus':
            beta = kwargs.get('beta') if kwargs.get('beta') else 1
            threshold = kwargs.get('threshold') if kwargs.get('threshold') else 20
            self.act_f = nn.Softplus(beta=beta, threshold=threshold)
        elif act_fun.lower() == 'softmax':
            dim = kwargs.get('dim') if kwargs.get('dim') else None
            self.act_f = nn.Softmax(dim=dim)
        elif act_fun.lower() == 'none':
            self.act_f = nn.Sequential()
        else:
            raise ValueError()

    def forward(self, x):
        if self.conv_first:
            x = self.conv(x)
            x = self.bn(x)
            x = self.act_f(x)
        else:
            x = self.bn(x)
            x = self.act_f(x)
            x = self.conv(x)
        return x

class residual(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, act_fun='relu', sn=False, bn=True, bias=False):
        super(residual, self).__init__()
        self.conv1 = convolution2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv2 = convolution2d(out_planes, out_planes, kernel_size=kernel_size, padding=kernel_size//2, stride=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        if stride != 1 or in_planes != out_planes:
            self.skip = convolution2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, sn=sn, bn=bn, act_fun='none')
        else:
            self.skip = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        skip = self.skip(x)
        return self.relu(conv2 + skip)

def sigmoid(x, min=1e-4, max=1-1e-4):
    x = torch.clamp(x.sigmoid_(), min=min, max=max)
    return x

class hswish(nn.Module):
    # used in mobilenetv3
    def __init__(self):
        super(hswish, self).__init__()

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    # used in mobilenetv3
    def __init__(self):
        super(hsigmoid, self).__init__()

    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

def test():
    conv1 = convolution2d(64, 128, 3, eps=1)
    resid = residual(64, 128, 7, 2)
    x = torch.Tensor(1, 64, 100, 100).uniform_()
    y = resid(x)
    print(x.size(), y.size())

if __name__ == '__main__':
    test()
