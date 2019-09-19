import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from block.DenseBlock import DenseBlock
from basic import convolution2d

class TransitionLayer(nn.Module):
    def __init__(self, inp_planes, out_planes, pool=False, bias=False, sn=False, bn=True, act_fun='relu'):
        super(TransitionLayer, self).__init__()
        self.conv = convolution2d(inp_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True) if pool else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        return (x, self.pool(x))

class DenseSupervision1(nn.Module):
    def __init__(self, inp_planes, out_planes, bias=False, sn=False, bn=True, act_fun='relu'):
        super(DenseSupervision1, self).__init__()
        self.right = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            convolution2d(inp_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False)
        )

    def forward(self, x1, x2):
        right = self.right(x1)
        return torch.cat([x2, right], 1)

class DenseSupervision(nn.Module):
    def __init__(self, inp_planes, out_planes=128, bias=False, sn=False, bn=True, act_fun='relu'):
        super(DenseSupervision, self).__init__()
        self.left = nn.Sequential(
            nn.MaxPool2d(2, 2, ceil_mode=True),
            convolution2d(inp_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False),
        )
        self.right = nn.Sequential(
            convolution2d(inp_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False),
            convolution2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun, conv_first=False)
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left, right], 1)
    

class DenseNet(nn.Module):
    def __init__(self, output='classifier', pre=None, bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        super(DenseNet, self).__init__()
        self.output = output
        if pre is None:
            self.pre = nn.Sequential(
                convolution2d(3, 64, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
                convolution2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
                convolution2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
                nn.MaxPool2d(2, 2, ceil_mode=True)
            )
            self.pre_channels = 128
        else:
            self.pre = pre
            self.pre_channels = kwargs.get('pre_channels')

        self.dense1 = DenseBlock(inp_planes=self.pre_channels, mid_planes=192, growth_rate=48, layer_num=6, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        n_channels1 = self.pre_channels + 6 * 48
        self.trans1 = TransitionLayer(n_channels1, n_channels1, pool=True, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dense2 = DenseBlock(inp_planes=n_channels1, mid_planes=192, growth_rate=48, layer_num=8, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        n_channels2 = n_channels1 + 8 * 48
        self.trans2 = TransitionLayer(n_channels2, n_channels2, pool=True, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dense3 = DenseBlock(inp_planes=n_channels2, mid_planes=192, growth_rate=48, layer_num=8, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        n_channels3 = n_channels2 + 8 * 48
        self.trans3 = TransitionLayer(n_channels3, n_channels3, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dense4 = DenseBlock(inp_planes=n_channels3, mid_planes=192, growth_rate=48, layer_num=8, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        n_channels4 = n_channels3 + 8 * 48
        self.trans4 = TransitionLayer(n_channels4, 256, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        self.dense_sup1 = DenseSupervision1(n_channels2, 256, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dense_sup2 = DenseSupervision(512, 256, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dense_sup3 = DenseSupervision(512, 128, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dense_sup4 = DenseSupervision(256, 128, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.dnese_sup5 = DenseSupervision(256, 128, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

        if self.output == 'classifier':
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            self.pool1 = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(256, 1024)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.dense1(x)
        _, x = self.trans1(x)
        x = self.dense2(x)
        f1, x = self.trans2(x)
        x = self.dense3(x)
        _, x = self.trans3(x)
        x = self.dense4(x)
        _, x = self.trans4(x)

        f2 = self.dense_sup1(f1, x)
        f3 = self.dense_sup2(f2)
        f4 = self.dense_sup3(f3)
        f5 = self.dense_sup4(f4)
        f6 = self.dnese_sup5(f5)

        if self.output == 'feature':
            return f1, f2, f3, f4, f5, f6
        elif self.output == 'classifier':
            y = self.pool1(f6)
            y = y.view(y.size(0), -1)
            y = self.relu(y)
            y = self.relu(self.fc1(y))
            y = self.fc2(y)
            return y

if __name__ == '__main__':
    net = DenseNet(output='feature', sn=True)
    x = torch.Tensor(2, 3, 320, 320).uniform_()
    outs = net(x)
    print([item.size() for item in outs])
