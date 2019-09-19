import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic import hsigmoid
from basic import SwitchNorm2d

class SeBlock(nn.Module):
    """SeBlock used in mobilenetv3, there is a litter difference between origin SeBlock
    Args:
        inp_planes: the number of input feature channels
        reduction: reduction
    Inputs: 
        x: Tensor, [bs * c * h * w]
    Outputs:
        return: Tensor, [bs * c * h * w]
    """
    def __init__(self, inp_planes, reduction=4, bias=False, sn=False, bn=True):
        super(SeBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp_planes, inp_planes // reduction, kernel_size=1, stride=1, padding=0, bias=bias),
            SwitchNorm2d(inp_planes // reduction, using_bn=bn) if sn else (nn.BatchNorm2d(inp_planes // reduction) if bn else nn.Sequential()),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp_planes // reduction, inp_planes, kernel_size=1, stride=1, padding=0, bias=bias),
            SwitchNorm2d(inp_planes, using_bn=bn) if sn else (nn.BatchNorm2d(inp_planes) if bn else nn.Sequential()),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MobileNetV3Block(nn.Module):
    def __init__(self, kernel_size, inp_planes, expand_planes, out_planes, nolinear, semodule, stride, bias=False, sn=False, bn=True):
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(inp_planes, expand_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = SwitchNorm2d(expand_planes, using_bn=bn) if sn else (nn.BatchNorm2d(expand_planes) if bn else nn.Sequential())
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_planes, expand_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_planes, bias=bias)
        self.bn2 = SwitchNorm2d(expand_planes, using_bn=bn) if sn else (nn.BatchNorm2d(expand_planes) if bn else nn.Sequential())
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn3 = SwitchNorm2d(out_planes, using_bn=bn) if sn else (nn.BatchNorm2d(out_planes) if bn else nn.Sequential())
        self.shortcut = nn.Sequential()
        if stride == 1 and inp_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias),
                SwitchNorm2d(out_planes, using_bn=bn) if sn else (nn.BatchNorm2d(out_planes) if bn else nn.Sequential()),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MobileNetV2Block(nn.Module):#MobileNet_2 网络的Bottleneck层
    n=0
    def __init__(self, in_planes, expansion, out_planes, repeat_times, stride , bias=False, sn=False, bn=True):
        super(MobileNetV2Block, self).__init__()
        inner_channels = in_planes*expansion
        #Bottlencek3个组件之一:'1*1-conv2d'
        self.conv1      = nn.Conv2d(in_planes, inner_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1        = SwitchNorm2d(inner_channels, using_bn=bn) if sn else (nn.BatchNorm2d(inner_channels) if bn else nn.Sequential())
        #Bottlencek3个组件之二:dwise
        self.conv2_with_stride = nn.Conv2d(inner_channels,inner_channels,kernel_size=3, stride=stride, padding=1, groups=inner_channels, bias=bias) #layer==1 stride=s
        self.conv2_no_stride   = nn.Conv2d(inner_channels,inner_channels,kernel_size=3, stride=1,      padding=1, groups=inner_channels, bias=bias) #layer>1  stride=1
        #Bottlencek3个组件之三:linear-1*1-conv2d'
        self.conv3 = nn.Conv2d(inner_channels, out_planes, kernel_size=1, stride=1, padding=0, groups=1, bias=bias)
        #当某个bottleneck重复出现时，'1*1-conv2d'的输入输出的通道数发生变化，不能再使用conv1了
        self.conv_inner = nn.Conv2d(out_planes, expansion*out_planes, kernel_size=1, stride=1, padding=0, bias=bias)
        #当某个bottleneck重复出现时，dwise的输入输出的通道数发生变化，不能再使用conv2_with_stride和conv2_no_stride了
        self.conv_inner_with_stride = nn.Conv2d(expansion*out_planes,expansion*out_planes,kernel_size=3, stride=stride, padding=1, groups=out_planes, bias=bias) #layer==1 stride=s
        self.conv_inner_no_stride   = nn.Conv2d(expansion*out_planes,expansion*out_planes,kernel_size=3, stride=1,         padding=1, groups=out_planes, bias=bias) #layer>1  stride=1
        #当某个bottleneck重复出现时，'linear-1*1-conv2d'的输入输出的通道数发生变化，不能再使用了
        self.conv3_inner = nn.Conv2d(expansion*out_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=1, bias=bias)
        #当某个bottleneck重复出现时，batchnorm的通道数也同样发生了变化
        self.bn_inner = SwitchNorm2d(expansion*out_planes, using_bn=bn) if sn else (nn.BatchNorm2d(expansion*out_planes) if bn else nn.Sequential())
        self.bn2 = SwitchNorm2d(out_planes, using_bn=bn) if sn else (nn.BatchNorm2d(out_planes) if bn else nn.Sequential())
        self.n = repeat_times

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn1(self.conv2_with_stride(out)))
        out = self.conv3(out)
        out = self.bn2(out)
        count = 2
        while(count<=self.n):
            temp = out
            out = F.relu6(self.bn_inner(self.conv_inner(out)))
            out = F.relu6(self.bn_inner(self.conv_inner_no_stride(out)))
            out = self.conv3_inner(out)
            out = self.bn2(out)
            out = out + temp
            count = count+1
        return out
