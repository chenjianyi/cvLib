import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic import convolution2d
from basic import L2Norm
from net import VGG16

class VGG16Extractor(nn.Module):
    def __init__(self, _type, bias=False, sn=False, bn=False, act_fun='relu'):
        super(VGG16Extractor, self).__init__()
        self._type = _type

        self.features = VGG16(output='feature', bias=bias, sn=sn, bn=bn, act_fun=act_fun, feature_levels=[12])
        self.norm4 = L2Norm(512, 20)

        self.conv5_1 = convolution2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv5_2 = convolution2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv5_3 = convolution2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.conv6 = convolution2d(512, 1024, kernel_size=3, padding=6, dilation=6, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv7 = convolution2d(1024, 1024, kernel_size=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.conv8_1 = convolution2d(1024, 256, kernel_size=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv8_2 = convolution2d(256, 512, kernel_size=3, padding=1, stride=2, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.conv9_1 = convolution2d(512, 128, kernel_size=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv9_2 = convolution2d(128, 256, kernel_size=3, padding=1, stride=2, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.conv10_1 = convolution2d(256, 128, kernel_size=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv10_2 = convolution2d(128, 256, kernel_size=3, padding=1, stride=2, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.conv11_1 = convolution2d(256, 128, kernel_size=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv11_2 = convolution2d(128, 256, kernel_size=3, padding=1, stride=2, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.conv12_1 = convolution2d(256, 128, kernel_size=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.conv12_2 = convolution2d(128, 256, kernel_size=4, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

    def forward(self, x):
        hs = []
        h = self.features(x)[0]
        hs.append(self.norm4(h))  # conv4_3

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = self.conv6(h)
        h = self.conv7(h)
        hs.append(h)  # conv7

        h = self.conv8_1(h)
        h = self.conv8_2(h)
        hs.append(h)  # conv8_2

        h = self.conv9_1(h)
        h = self.conv9_2(h)
        hs.append(h)  # conv9_2

        h = self.conv10_1(h)
        h = self.conv10_2(h)
        hs.append(h)  # conv10_2

        h = self.conv11_1(h)
        h = self.conv11_2(h)
        hs.append(h)  # conv11_2
        if self._type == 'ssd300':
            return hs
        elif self._type == 'ssd512':
            h = self.conv12_1(h)
            h = self.conv12_2(h)
            hs.append(h)  # conv12_2
            return hs

def VGG_SSD300(output='feature', bias=False, sn=False, bn=False, act_fun='relu'):
    return VGG16Extractor(_type='ssd300', bias=bias, sn=sn, bn=bn, act_fun=act_fun)

def VGG_SSD512(output='feature', bias=False, sn=False, bn=False, act_fun='relu'):
    return VGG16Extractor(_type='ssd512', bias=bias, sn=sn, bn=bn, act_fun=act_fun)

if __name__ == '__main__':
    net = VGG_SSD512()
    outs = net(torch.randn(1,3,320, 512))
    print([item.size() for item in outs])
