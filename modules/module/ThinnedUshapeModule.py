import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic import convolution2d

class TUM(nn.Module):
    def __init__(self, first_level=True, input_planes=128, is_smooth=True, side_channel=512, scales=6, bias=False, sn=False, bn=True, act_fun='relu'):
        super(TUM, self).__init__()
        self.is_smooth = is_smooth
        self.side_channel = side_channel
        self.input_planes = input_planes
        self.planes = 2 * self.input_planes
        self.first_level = first_level
        self.scales = scales
        self.in1 = input_planes + side_channel if not first_level else input_planes

        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)), convolution2d(self.in1, self.planes, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun))

        for i in range(self.scales - 2):
            if not i == self.scales - 3:
                self.layers.add_module(
                    '{}'.format(len(self.layers)), convolution2d(self.planes, self.planes, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
                )
            else:
                self.layers.add_module(
                    '{}'.format(len(self.layers)), convolution2d(self.planes, self.planes, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun) #chenjianyi
                )
        self.toplayer = nn.Sequential(convolution2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun))
        self.latlayer = nn.Sequential()
        for i in range(self.scales - 2):
            self.latlayer.add_module(
                '{}'.format(len(self.latlayer)), convolution2d(self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
            )
        self.latlayer.add_module('{}'.format(len(self.latlayer)), convolution2d(self.in1, self.planes, kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun))

        if self.is_smooth:
            smooth = list()
            for i in range(self.scales - 1):
                smooth.append(convolution2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun))
            self.smooth = nn.Sequential(*smooth)

    def _upsample_add(self, x, y, fuse_type='nearest'):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode=fuse_type) + y

    def forward(self, x, y):
        if not self.first_level:
            x = torch.cat([x, y], 1)

        conved_feat = [x]
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            conved_feat.append(x)

        deconved_feat = [self.toplayer[0](conved_feat[-1])]
        for i in range(len(self.latlayer)):
            deconved_feat.append(
                self._upsample_add(deconved_feat[i], self.latlayer[i](conved_feat[len(self.layers) - i - 1]))
            )

        if self.is_smooth:
            smoothed_feat = [deconved_feat[0]]
            for i in range(len(self.smooth)):
                smoothed_feat.append(self.smooth[i](deconved_feat[i + 1]))
            return smoothed_feat

        return deconved_feat


class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        '''Scale-wise Feature Aggregation Module
        Params:
            planes:
            num_levels:
            num_scales:
            compress_ratio:
        '''
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList(
            [nn.Conv2d(self.planes * self.num_levels, self.planes * self.num_levels // 16, kernel_size=1, stride=1, padding=0)] * self.num_scales
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList(
            [nn.Conv2d(self.planes * self.num_levels // 16, self.planes * self.num_levels, kernel_size=1, stride=1, padding=0)] * self.num_scales
        )
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf * _tmp_f)
        return attention_feat
