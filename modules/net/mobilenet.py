import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from block import MobileNetV3Block, SeBlock, MobileNetV2Block
from basic import hswish
from basic import SwitchNorm2d, SwitchNorm1d

class MobileNetV3_Large(nn.Module):
    def __init__(self, output='classifier', bias=False, sn=False, bn=True, **kwargs):
        super(MobileNetV3_Large, self).__init__()
        self.output = output

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=bias)
        self.bn1 = SwitchNorm2d(16, using_bn=bn) if sn else (nn.BatchNorm2d(16) if bn else nn.Sequential())
        self.hs1 = hswish()

        self.bneck = nn.ModuleList([
            # MobileNetV3Block(kernel_size, inp_planes, expand_planes, out_planes, nolinear, semodule, stride)
            MobileNetV3Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeBlock(40, bias=bias, sn=sn, bn=bn), 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeBlock(40, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeBlock(40, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 40, 240, 80, hswish(), None, 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 80, 200, 80, hswish(), None, 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 80, 184, 80, hswish(), None, 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 80, 184, 80, hswish(), None, 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 80, 480, 112, hswish(), SeBlock(112), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 112, 672, 112, hswish(), SeBlock(112), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 112, 672, 160, hswish(), SeBlock(160), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 160, 672, 160, hswish(), SeBlock(160), 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 160, 960, 160, hswish(), SeBlock(160), 1, bias=bias, sn=sn, bn=bn),
        ])

        if output == 'classifier':
            self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=bias)
            self.bn2 = SwitchNorm2d(960, using_bn=bn) if sn else (nn.BatchNorm2d(960) if bn else nn.Sequential())
            self.hs2 = hswish()
            self.linear3 = nn.Linear(960, 1280)
            self.bn3 = SwitchNorm1d(1280, using_bn=bn) if sn else (nn.BatchNorm1d(1280) if bn else nn.Sequential())
            self.hs3 = hswish()
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            self.linear4 = nn.Linear(1280, num_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        x = self.hs1(self.bn1(self.conv1(x)))
        for i, bneck in enumerate(self.bneck):
            x = bneck(x)
            if i in [2, 5, 12, 14]:
                outs.append(x)
        if self.output == 'feature':
            return outs
        if self.output == 'classifier':
            x = self.hs2(self.bn2(self.conv2(x)))
            x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.view(x.size(0), -1)
            x = self.hs3(self.bn3(self.linear3(x)))
            x = self.linear4(x)
            return x

class MobileNetV3_Small(nn.Module):
    def __init__(self, output='classifier', bias=False, sn=False, bn=True, **kwargs):
        super(MobileNetV3_Small, self).__init__()
        self.output = output

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=bias)
        self.bn1 = SwitchNorm2d(16, using_bn=bn) if sn else (nn.BatchNorm2d(16) if bn else nn.Sequential())
        self.hs1 = hswish()

        self.bneck = nn.ModuleList([
            # MobileNetV3Block(kernel_size, inp_planes, expand_planes, out_planes, nolinear, semodule, stride)
            MobileNetV3Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeBlock(16, bias=bias, sn=sn, bn=bn), 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 24, 96, 40, hswish(), SeBlock(40, bias=bias, sn=sn, bn=bn), 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 40, 240, 40, hswish(), SeBlock(40, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 40, 240, 40, hswish(), SeBlock(40, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 40, 120, 48, hswish(), SeBlock(48, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 48, 144, 48, hswish(), SeBlock(48, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 48, 288, 96, hswish(), SeBlock(96, bias=bias, sn=sn, bn=bn), 2, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 96, 576, 96, hswish(), SeBlock(96, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
            MobileNetV3Block(5, 96, 576, 96, hswish(), SeBlock(96, bias=bias, sn=sn, bn=bn), 1, bias=bias, sn=sn, bn=bn),
        ])

        if output == 'classifier':
            self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=bias)
            self.bn2 = SwitchNorm2d(576, using_bn=bn) if sn else (nn.BatchNorm2d(576) if bn else nn.Sequential())
            self.hs2 = hswish()
            self.linear3 = nn.Linear(576, 1280)
            self.bn3 = SwitchNorm1d(1280, using_bn=bn) if sn else (nn.BatchNorm1d(1280) if bn else nn.Sequential())
            self.hs3 = hswish()
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            self.linear4 = nn.Linear(1280, num_classes)

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        x = self.hs1(self.bn1(self.conv1(x)))
        for i, bneck in enumerate(self.bneck):
            x = bneck(x)
            if i in [0, 2, 7, 10]:
                outs.append(x)
        if self.output == 'feature':
            return outs
        if self.output == 'classifier':
            x = self.hs2(self.bn2(self.conv2(x)))
            x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.view(x.size(0), -1)
            x = self.hs3(self.bn3(self.linear3(x)))
            x = self.linear4(x)
            return x

class MobileNetV2(nn.Module):
    #[input_channels, t, c, n, s] 论文中的参数列表
    param = [[ 32, 1,  16, 1, 1], 
             [ 16, 6,  24, 2, 2],  
             [ 24, 6,  32, 3, 2],  
             [ 32, 6,  64, 4, 2],  
             [ 64, 6,  96, 3, 1],  
             [ 96, 6, 160, 3, 2],  
             [160, 6, 320, 1, 1]]
    def __init__(self, output='classifier', bias=False, sn=False, bn=True, **kwargs):
        super(MobileNetV2,self).__init__()
        self.output = output
        self.bias, self.sn, self.bn = bias, sn, bn
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=bias),
            SwitchNorm2d(32, using_bn=bn) if sn else (nn.BatchNorm2d(32) if bn else nn.Sequential()),
            nn.ReLU6()
        )
        self.bnecks = self._make_layers()
        if output == 'classifier':
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            self.conv_1280 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=bias)
            self.bn_1280 = SwitchNorm2d(1280, using_bn=bn) if sn else (nn.BatchNorm2d(1280) if bn else nn.Sequential())
            self.conv_end = nn.Linear(1280, num_classes)
        self.init_params()

    def _make_layers(self):
        layer = nn.ModuleList()
        for i in range( len(self.param) ):
            layer.append(MobileNetV2Block(self.param[i][0], self.param[i][1], self.param[i][2], self.param[i][3], self.param[i][4], bias=self.bias, sn=self.sn, bn=self.bn))
        return layer

    def forward(self, x):
        x = self.pre(x)
        outs = []
        for i, bneck in enumerate(self.bnecks):
            x = bneck(x)
            if i in [1, 2, 3, 5]:
                outs.append(x)
        if self.output == 'feature':
            return outs
        x = nn.ReLU6(inplace=True)(self.bn_1280(self.conv_1280(x)))
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        x = self.conv_end(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def test():
    net = MobileNetV3_Large(bias=True, sn=True, bn=True, output='feature')
    x = torch.Tensor(2, 3, 416, 416).uniform_()
    outs = net(x)
    for out in outs:
        print(out.size())

if __name__ == '__main__':
    test()
