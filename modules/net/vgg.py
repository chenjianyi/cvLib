import sys
import os
#sys.path.append('..')
#sys.path.append('../..')
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn

from basic import convolution2d

class VGG(nn.Module):
    def __init__(self, cfg, i=3, output='classifier', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        super(VGG, self).__init__()
        layers = []
        in_channels = i
        self.output = output
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = convolution2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
                layers += [conv2d]
                in_channels = v
                """
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
                if with_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                """
        if output == 'feature':
            self.layers = nn.ModuleList(layers)
            self.feature_levels = kwargs.get('feature_levels') if kwargs.get('feature_levels') else [len(self.layers) - 1]
        if output == 'classifier':
            cls_layers = []
            cls_layers += [nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout()]
            cls_layers += [nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout()]
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            cls_layers += [nn.Linear(4096, num_classes)]
            self.cls_layers = nn.ModuleList(cls_layers)
            self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if self.output == 'feature':
            out = []
            for k in range(len(self.layers)):
                x = self.layers[k](x)
                if k in self.feature_levels:
                    out.append(x)
            return out
        elif self.output == 'classifier':
            for k in range(len(self.layers)):
                x = self.layers[k](x)
            x = x.view(-1, 512 * 7 * 7)
            for k in range(len(self.cls_layers)):
                x = self.cls_layers[k](x)
            out = x
            return out

class VGG16(VGG):
    def __init__(self, i=3, output='classifier', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'M']
        super(VGG16, self).__init__(base, i, output, bias=bias, sn=sn, bn=bn, act_fun=act_fun, **kwargs)


def test():
    net = VGG16(3, output='feature').cuda()
    x = torch.Tensor(1,3,224,224).cuda()
    x = net(x)
    for xx in x:
        print(xx.size())

if __name__ == '__main__':
    test()
