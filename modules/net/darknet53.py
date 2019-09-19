import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
from torch import nn

from basic import convolution2d, residual

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, inp_planes):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(inp_planes/2)

        self.layer1 = conv_batch(inp_planes, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, inp_planes)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(nn.Module):
    def __init__(self, block=residual, output='classifier', bias=False, sn=False, bn=True, **kwargs):
        super(Darknet53, self).__init__()

        self.output = output
        self.num_classes = kwargs.get("num_classes") if kwargs.get("num_classes") else 1000

        self.conv1 = convolution2d(3, 32, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn)
        self.conv2 = convolution2d(32, 64, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn)
        self.residual_block1 = self.make_layer(block, inp_planes=64, out_planes=64, num_blocks=1, bias=bias, sn=sn, bn=bn)
        self.conv3 = convolution2d(64, 128, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn)
        self.residual_block2 = self.make_layer(block, inp_planes=128, out_planes=128, num_blocks=2, bias=bias, sn=sn, bn=bn)
        self.conv4 = convolution2d(128, 256, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn)
        self.residual_block3 = self.make_layer(block, inp_planes=256, out_planes=256, num_blocks=8, bias=bias, sn=sn, bn=bn)
        self.conv5 = convolution2d(256, 512, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn)
        self.residual_block4 = self.make_layer(block, inp_planes=512, out_planes=512, num_blocks=8, bias=bias, sn=sn, bn=bn)
        self.conv6 = convolution2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn)
        self.residual_block5 = self.make_layer(block, inp_planes=1024, out_planes=1024, num_blocks=4, bias=bias, sn=sn, bn=bn)

        if output == 'classifier':
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        outs = []
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        outs.append(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        outs.append(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        outs.append(out)

        if self.output == 'feature':
            return outs
        elif self.output == 'classifier':
            out = self.global_avg_pool(out)
            out = out.view(-1, 1024)
            out = self.fc(out)
            return out


    def make_layer(self, block, inp_planes, out_planes, num_blocks, kernel_size=3, stride=1, padding=1, act_fun='leakyrelu', sn=False, bn=True, bias=False):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(inp_planes, out_planes, kernel_size=kernel_size, stride=stride, act_fun=act_fun, sn=sn, bn=bn, bias=bias))
        return nn.Sequential(*layers)


def test():
    net = Darknet53(residual, num_classes=1000, output='feature').cuda()
    x = torch.Tensor(2, 3, 2400, 2400).uniform_().cuda()
    outs = net(x)
    print([out.size() for out in outs])

if __name__ == '__main__':
    test()
