import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from basic import convolution2d
from module import TUM, SFAM
from vgg import VGG16


def get_backbone(backbone_name='vgg16', bias=False, sn=False, bn=True, act_fun='relu'):
    return VGG16(3, with_bn=False, output='feature', feature_levels=[12], bias=bias, sn=sn, bn=bn, act_fun=act_fun)

class M2Det(nn.Module):
    def __init__(self, num_scales, num_levels, planes, backbone='vgg16', smooth=True, sfam=True, output='feature', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        '''M2Det: Multi-level Multi-scale single-shot object Detector
        Params:
        '''
        super(M2Det, self).__init__()
        #self.size = size
        self.bias, self.sn, self.bn, self.act_fun = bias, sn, bn, act_fun
        self.backbone = backbone
        self.num_scales = num_scales
        self.num_levels = num_levels
        self.planes = planes
        self.smooth = smooth
        self.sfam = sfam
        self.output = output
        self.num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 80
        self.construct_modules()

    def construct_modules(self):
        # construct tums
        for i in range(self.num_levels):
            if i == 0:
                setattr(self, 'unet{}'.format(i+1), \
                        TUM(first_level=True, input_planes=self.planes//2, is_smooth=self.smooth, scales=self.num_scales, side_channel=512, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun))
            else:
                setattr(self, 'unet{}'.format(i+1), \
                        TUM(first_level=False, input_planes=self.planes//2, is_smooth=self.smooth, scales=self.num_scales, side_channel=self.planes, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun))

        
        # construct backbone
        self.base1 = get_backbone(self.backbone, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun)
        self.base2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        if self.backbone == 'vgg16':
            shallow_in, shallow_out = 512, 256
            deep_in, deep_out = 1024, 512
        self.reduce = convolution2d(shallow_in, shallow_out, kernel_size=3, stride=1, padding=1, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun)
        self.up_reduce = convolution2d(deep_in, deep_out, kernel_size=1, stride=1, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun)
        
        # construct others
        self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256 * self.num_levels)
        self.leach = nn.ModuleList(
            [convolution2d(deep_out+shallow_out, self.planes//2, kernel_size=(1, 1), stride=(1, 1), bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun)] * self.num_levels
        )

        # construct localization and recognition layers
        loc_ = list()
        conf_ = list()
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(self.planes*self.num_levels, 4*6, kernel_size=3, stride=1, padding=1))
            conf_.append(nn.Conv2d(self.planes*self.num_levels, self.num_classes*6, kernel_size=3, stride=1, padding=1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        # construct SFAM module
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)
        
    def forward(self, x):
        base_feats = self.base1(x)
        base_feats.append(self.base2(base_feats[0]))
        base_feature_size = (base_feats[0].size(2), base_feats[0].size(3))
        base_feature = torch.cat(
            (self.reduce(base_feats[0]), F.interpolate(self.up_reduce(base_feats[1]), size=base_feature_size, mode='nearest')), 1
        )

        # tum_outs: multi-level multi-scale feature
        tum_outs = [getattr(self, 'unet{}'.format(1))(self.leach[0](base_feature), 'none')]
        for i in range(1, self.num_levels, 1):
            tum_outs.append(
                getattr(self, 'unet{}'.format(i+1))(self.leach[i](base_feature), tum_outs[i-1][-1])
            )

        # concat with same scales
        sources = [torch.cat([_fx[i-1] for _fx in tum_outs], 1) for i in range(self.num_scales, 0, -1)]

        # forward_sfam
        if self.sfam:
            sources = self.sfam_module(sources)
        sources[0] = self.Norm(sources[0])

        if self.output == 'feature':
            return sources

        loc, conf = [], []
        for (xx, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(xx).permute(0, 2, 3, 1).contiguous())
            conf.append(c(xx).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.training:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), self.num_classes))
            )
        return output

def test():
    #from tools import summary
    conf = {'backbone': 'vgg16', 'num_scales': 3, 'num_levels': 3, 'planes': 256, 'num_classes': 2, 'smooth': True, 'sfam': True}
    net = M2Det(**conf).cuda()
    x = torch.Tensor(1, 3, 500, 500).uniform_().cuda()
    print(net(x))
    summary(net, (3, 416, 416))

if __name__ == '__main__':
    test()
    
