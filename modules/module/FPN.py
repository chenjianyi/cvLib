import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from basic import convolution2d
from basic import hswish
from block import MobileNetV3Block

class FPNModule(nn.Module):
    """FPN Module: merge features of two scales
    """
    def __init__(self, inp_planes, pre_planes, pre_size, post_planes, post_size, merge_type='sum', bias=True, sn=False, bn=True, act_fun='leakyrelu'):
        super(FPNModule, self).__init__()
        self.merge_type = merge_type
        self.pre = convolution2d(inp_planes, pre_planes, kernel_size=pre_size, stride=1, padding=pre_size//2, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        self.post = nn.ModuleList([
             convolution2d(post_planes[i-1], post_planes[i], kernel_size=post_size[i], stride=1, padding=post_size[i]//2, bias=bias, sn=sn, bn=bn, act_fun=act_fun) \
                for i in range(1, len(post_planes))
        ])
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x1 = self.pre(x1)
        x1 = self.upsample(x1)
        if self.merge_type == 'sum':
            x = x1 + x2
        elif self.merge_type == 'concat':
            x = torch.cat([x1, x2], 1)
        for i, module in enumerate(self.post):
            x = module(x)
        return x

class FPNYoloV3(nn.Module):
      """FPN used for YoloV3
      """
      pre_planes = [512, 256]
      pre_size = [1, 1]
      post_planes = [[1024, 512, 256, 512, 256, 512],
                     [512, 256, 128, 256, 128, 256]]
      post_size = [[1, 3, 1, 3, 1, 3],
                   [1, 3, 1, 3, 1, 3]]
      def __init__(self, feature_planes, bias=True, sn=False, bn=True, act_fun='leakyrelu'):
          super(FPNYoloV3, self).__init__()
          self.FPN_Modules = nn.ModuleList()
          for i in range(0, len(feature_planes) - 1):
              self.FPN_Modules.append(FPNModule(feature_planes[-i-1], self.pre_planes[i], self.pre_size[i], self.post_planes[i], self.post_size[i], merge_type='concat', \
                                                bias=bias, sn=sn, bn=bn, act_fun=act_fun))
          self.last = convolution2d(feature_planes[-1], feature_planes[-1], kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

      def forward(self, x_list):
          [x1, x2, x3] = x_list
          x2 = self.FPN_Modules[0](x3, x2)
          x1 = self.FPN_Modules[1](x2, x1)
          x3 = self.last(x3)
          return x1, x2, x3

class FPNMobileNetV3Module(nn.Module):
      def __init__(self, inp_planes, pre_planes, pre_size, post_planes, post_size, merge_type='sum', bias=True, sn=False, bn=True, act_fun='leakyrelu'):
          super(FPNMobileNetV3Module, self).__init__()
          self.merge_type = merge_type
          self.pre = MobileNetV3Block(pre_size, inp_planes, inp_planes, pre_planes, hswish(), None, 1, bias=bias, sn=sn, bn=bn)
          self.post = nn.ModuleList([
              MobileNetV3Block(post_size[i], post_planes[i-1], post_planes[i-1] * 2, post_planes[i], hswish(), None, 1, bias=bias, sn=sn, bn=bn) for i in range(1, len(post_planes))
          ])
          self.upsample = nn.Upsample(scale_factor=2)

      def forward(self, x1, x2):
          x1 = self.pre(x1)
          x1 = self.upsample(x1)
          if self.merge_type == 'sum':
              x = x1 + x2
          elif self.merge_type == 'concat':
              x = torch.cat([x1, x2], 1)
          for module in self.post:
              x = module(x)
          return x

class FPNMobileNetV3(nn.Module):
      """FPN using MobileNetV3Module
      """
      pre_planes = [80, 80]
      pre_size = [3, 3]
      post_planes = [[240, 320, 160],
                     [120, 240, 40]]
      post_size = [[3, 3, 3],
                   [3, 3, 3]]

      def __init__(self, feature_planes, bias=True, sn=False, bn=True):
          super(FPNMobileNetV3, self).__init__()
          self.FPN_modules = nn.ModuleList()
          for i in range(0, len(feature_planes) - 1):
              self.FPN_modules.append(FPNMobileNetV3Module(feature_planes[-i-1], self.pre_planes[i], self.pre_size[i], self.post_planes[i], self.post_size[i], merge_type='concat', bias=bias, sn=sn, bn=bn))
          self.last = MobileNetV3Block(3, feature_planes[-1], feature_planes[-1] * 2, feature_planes[-1], hswish(), None, 1, bias=bias, sn=sn, bn=bn)

      def forward(self, x_list):
          [x1, x2, x3] = x_list
          x2 = self.FPN_modules[0](x3, x2)
          x1 = self.FPN_modules[1](x2, x1)
          x3 = self.last(x3)
          return x1, x2, x3
