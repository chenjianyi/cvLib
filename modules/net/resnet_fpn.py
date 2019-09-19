import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from block import Bottleneck

class Resnet_FPN(nn.Module):
    def __init__(self, block, num_blocks, output='feature', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        super(Resnet_FPN, self).__init__()
        self.output = output
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        if self.output == 'feature':
            # Top-down layers
            self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

            # Lateral layers
            self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
            self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

            # Smooth layers
            self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        elif self.output == 'classifier':
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(256, 1024)
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            self.fc2 = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p8 = self.conv8(F.relu(p7))
        p9 = self.conv9(F.relu(p8))
        # Top-down
        if self.output == 'feature':
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4))
            p3 = self._upsample_add(p4, self.latlayer2(c3))
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            return p3, p4, p5, p6, p7, p8, p9
        elif self.output == 'classifier':
            fc1 = self.pool(p9)
            fc1 = fc1.view(fc1.size(0), -1)
            fc2 = self.fc1(fc1)
            cls = self.fc2(fc2)
            return cls


def Resnet_FPN50(output='feature', bias=False, sn=False, bn=True, act_fun='relu'):
    return Resnet_FPN(Bottleneck, [3,4,6,3], output=output, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

def Resnet_FPN101(output='feature', bias=False, sn=False, bn=True, act_fun='relu'):
    return Resnet_FPN(Bottleneck, [3,4,23,3], output=output, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

def Resnet_FPN152(output='feature', bias=False, sn=False, bn=True, act_fun='relu'):
    return Resnet_FPN(Bottleneck, [3,8,36,3], output=output, bias=bias, sn=sn, bn=bn, act_fun=act_fun)


if __name__ == '__main__':
    net = Resnet_FPN152(output='feature')
    fms = net(torch.randn(1,3,320, 512))
    print([item.size() for item in fms])
