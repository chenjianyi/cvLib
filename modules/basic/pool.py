import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from basic import convolution2d
from ops._cpools import TopPool, BottomPool, LeftPool, RightPool

class pool_corner(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool_corner, self).__init__()
        self.p1_conv1 = convolution2d(dim, 128, kernel_size=3, stride=1, padding=1, bn=True)
        self.p2_conv1 = convolution2d(dim, 128, kernel_size=3, stride=1, padding=1, bn=True)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution2d(dim, dim, kernel_size=3, stride=1, padding=1, bn=True)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution2d(dim, 128, kernel_size=3, stride=1, padding=1, bn=True)
        self.look_conv2 = convolution2d(dim, 128, kernel_size=3, stride=1, padding=1, bn=True)
        self.p1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.p2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # x: bs * dim * h * w
        look_conv1 = self.look_conv1(x)  # bs * 128 * h * w
        p1_conv1 = self.p1_conv1(x)  # bs * 128 * h * w
        look2 = self.pool2(look_conv1)  # bs * 128 * h * w
        p1_look_conv = self.p1_look_conv(p1_conv1 + look2)  # bs * 128 * h * w
        pool1 = self.pool1(p1_look_conv)  # bs * 128 * h * w

        look_conv2 = self.look_conv2(x)  # bs * 128 * h * w
        p2_conv1 = self.p2_conv1(x)  # bs * 128 * h * w
        look1 = self.pool1(look_conv2)  # bs * 128 * h * w
        p2_look_conv = self.p2_look_conv(p2_conv1 + look1)  # bs * 128 * h * w
        pool2 = self.pool2(p2_look_conv)  # bs * 128 * h * w

        p_conv1 = self.p_conv1(pool1 + pool2)  # bs * dim * h * w
        p_bn1 = self.p_bn1(p_conv1)  # bs * dim * h * w

        conv1 = self.conv1(x)  # bs * dim * h * w
        bn1 = self.bn1(conv1)  # bs * dim * h * w
        relu1 = self.relu1(p_bn1 + bn1)  # bs * dim * h * w

        conv2 = self.conv2(relu1)  # bs * dim * h * w
        return conv2

class  pool_cross(nn.Module):
    def __init__(self, dim, pool1, pool2, pool3, pool4):
        super(pool_cross, self).__init__()
        self.p1_conv1 = convolution2d(dim, 128, kernel_size=3, stride=1, padding=1, bn=True)
        self.p2_conv1 = convolution2d(dim, 128, kernel_size=3, stride=1, padding=1, bn=True)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution2d(dim, dim, kernel_size=3, stride=1, padding=1, bn=True)

        self.pool1 = pool1()
        self.pool2 = pool2()
        self.pool3 = pool3()
        self.pool4 = pool4()

    def forward(self, x):
        # x: bs * dim * h * w
        p1_conv1 = self.p1_conv1(x)  # bs * 128 * h * w
        pool1 = self.pool1(p1_conv1)  # bs * 128 * h * w
        pool1_3 = self.pool3(pool1)  # bs * 128 * h * w

        p2_conv1 = self.p2_conv1(x)  # bs * 128 * h * w
        pool2 = self.pool2(p2_conv1)  # bs * 128 * h * w
        pool2_4 = self.pool4(pool2)  # bs * 128 * h * w

        p_conv1 = self.p_conv1(pool1_3 + pool2_4)  # bs * dim * h * w
        p_bn1 = self.p_bn1(p_conv1)  # bs * dim * h * w

        conv1 = self.conv1(x)  # bs * dim * h * w
        bn1 = self.bn1(conv1)  # bs * dim * h * w
        relu1 = self.relu1(p_bn1 + bn1)  # bs * dim * h * w

        conv2 = self.conv2(relu1)
        return conv2

class TopLeftPool(pool_corner):
    def __init__(self, dim):
        super(TopLeftPool, self).__init__(dim, TopPool, LeftPool)

class BottomRightPool(pool_corner):
    def __init__(self, dim):
        super(BottomRightPool, self).__init__(dim, BottomPool, RightPool)

class CenterPool(pool_cross):
    def __init__(self, dim):
        super(CenterPool, self).__init__(dim, TopPool, LeftPool, BottomPool, RightPool)
