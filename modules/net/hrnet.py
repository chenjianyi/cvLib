import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

from block import BasicBlock, Bottleneck
from module import HighResolutionModule
from basic import convolution2d

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_layer(block, inplanes, planes, blocks, stride=1, bias=False, sn=False, bn=True, act_fun='relu'):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = convolution2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0, bias=bias, sn=sn, bn=bn, act_fun='none')

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, bias=bias, sn=sn, bn=bn, act_fun=act_fun))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes, bias=bias, sn=sn, bn=bn, act_fun=act_fun))

    return nn.Sequential(*layers)

def _make_transition_layer(num_channels_pre_layer, num_channels_cur_layer, bias=False, sn=False, bn=True, act_fun='relu'):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    transition_layers = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                transition_layers.append(
                    convolution2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
                )
            else:
                transition_layers.append(nn.Sequential())
        else:
            conv3x3s = []
            for j in range(i + 1 - num_branches_pre):
                inchannels = num_channels_pre_layer[-1]
                outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                conv3x3s.append(
                    convolution2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
                )
            transition_layers.append(nn.Sequential(*conv3x3s))

    return nn.ModuleList(transition_layers)

def _make_stage(layer_config, num_inchannels, multi_scale_output=True, bias=False, sn=False, bn=True, act_fun='relu'):
    num_modules = layer_config['num_modules']
    num_branches = layer_config['num_branches']
    num_blocks = layer_config['num_blocks']
    num_channels = layer_config['num_channels']
    block = blocks_dict[layer_config['block']]
    fuse_method = layer_config['fuse_method']

    modules = []
    for i in range(num_modules):
        if not multi_scale_output and i == num_modules - 1:
            reset_multi_scale_output = False
        else:
            reset_multi_scale_output = True

        modules.append(HighResolutionModule(
            num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output, bias=bias, sn=sn, bn=bn, act_fun=act_fun
        ))
        num_inchannels = modules[-1].get_num_inchannels()

    return nn.Sequential(*modules), num_inchannels

def _make_head(pre_stage_channels, head_channels, bias=False, sn=False, bn=True, act_fun='relu'):
    head_block = Bottleneck
    incre_modules = []
    for i, channels in enumerate(pre_stage_channels):
        incre_module = _make_layer(head_block, channels, head_channels[i], 1, stride=1)
        incre_modules.append(incre_module)
    incre_modules = nn.ModuleList(incre_modules)

    downsample_modules = []
    for i in range(len(pre_stage_channels) - 1):
        in_channels = head_channels[i] * head_block.expansion
        out_channels = head_channels[i + 1] * head_block.expansion
        downsample_module = convolution2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
        
        downsample_modules.append(downsample_module)
    downsample_modules = nn.ModuleList(downsample_modules)

    final_layer = convolution2d(head_channels[-1] * head_block.expansion, 2048, kernel_size=1, stride=1, padding=0, bias=bias, sn=sn, bn=bn, act_fun=act_fun)

    return incre_modules, downsample_modules, final_layer
    

blocks_dict = {
    'BasicBlock': BasicBlock,
    'Bottleneck': Bottleneck,
}

class HighResolutionNet(nn.Module):
    def __init__(self, stages_config, pre=None, output='classifier', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
        super(HighResolutionNet, self).__init__()
        self.stages_config =  stages_config
        self.output = output
        if pre is None:
            self.pre = nn.Sequential(
                convolution2d(3, 64, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
                convolution2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias, sn=sn, bn=bn, act_fun=act_fun),
                _make_layer(Bottleneck, 64, 64, 4, bias=bias, sn=sn, bn=bn, act_fun=act_fun)
            )
            pre_stage_channels = [256]
        else:
            self.pre = pre
            pre_stage_channels = [kwargs.get('pre_channels')]

        ####
        self.body_list = nn.ModuleList()
        pre_stage_channels = [256]
        for stage_cfg in self.stages_config:
            num_channels = stage_cfg['num_channels']
            block = blocks_dict[stage_cfg['block']]
            num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
            transition = _make_transition_layer(pre_stage_channels, num_channels, sn=sn, bn=bn, act_fun=act_fun)
            stage, pre_stage_channels = _make_stage(stage_cfg, num_channels, sn=sn, bn=bn, act_fun=act_fun)
            self.body_list.append(nn.ModuleList([
                transition,
                stage
            ]))

        if self.output == 'classifier':
            head_channels = kwargs.get('head_channels') if kwargs.get('head_channels') else self.stages_config[-1]['num_channels']
            num_classes = kwargs.get('num_classes') if kwargs.get('num_classes') else 1000
            self.incre_modules, self.downsample_modules, self.final_layer = _make_head(pre_stage_channels, head_channels, sn=sn, bn=bn, act_fun=act_fun)
            self.classifer = nn.Linear(2048, num_classes)

    def forward(self, x):
        # pre
        x = self.pre(x)
        ###
        for i in range(len(self.stages_config)):
            x_list = []
            if i == 0:
                for j in range(self.stages_config[i]['num_branches']):
                    x_list.append(self.body_list[i][0][j](x))
            else:
                y_list.append(y_list[-1])
                for j in range(self.stages_config[i]['num_branches']):
                    x_list.append(self.body_list[i][0][j](y_list[j]))
            y_list = self.body_list[i][1](x_list)

        if self.output == 'feature':
            return y_list

        # head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsample_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsample_modules[i](y)
        y = self.final_layer(y)

        # classifier
        y = F.avg_pool2d(y, kernel_size=y.size()[2: ]).view(y.size(0), -1)
        y = self.classifer(y)

        return y

def HighResolutionNet_3stages(output='feature', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
    stage2 = {
        'num_modules': 1, 'num_branches': 2, 'num_blocks': [4, 4], 'num_channels': [32, 64], \
        'block': 'BasicBlock', 'fuse_method': 'sum'
    }
    stage3 = {
        'num_modules': 1, 'num_branches': 3, 'num_blocks': [4, 4, 4], 'num_channels': [32, 64, 128], \
        'block': 'BasicBlock', 'fuse_method': 'sum'
    }
    stages = [stage2, stage3]
    net = HighResolutionNet(stages, output=output)
    return net

def HighResolutionNet_4stages(output='feature', bias=False, sn=False, bn=True, act_fun='relu', **kwargs):
    stage2 = {
        'num_modules': 1, 'num_branches': 2, 'num_blocks': [4, 4], 'num_channels': [32, 64], \
        'block': 'BasicBlock', 'fuse_method': 'sum'
    }
    stage3 = {
        'num_modules': 1, 'num_branches': 3, 'num_blocks': [4, 4, 4], 'num_channels': [32, 64, 128], \
        'block': 'BasicBlock', 'fuse_method': 'sum'
    }
    stage4 = {
        'num_modules': 1, 'num_branches': 4, 'num_blocks': [4, 4, 4, 4], 'num_channels': [32, 64, 128, 256], \
        'block': 'BasicBlock', 'fuse_method': 'sum'
    }
    stages = [stage2, stage3, stage4]
    net = HighResolutionNet(stages, output=output, **kwargs)
    return net



def test():
    from tools import summary
    net = HighResolutionNet_3stages(output='feature', sn=True).cuda()
    x = torch.Tensor(1, 3, 208, 208).uniform_().cuda()
    y = net(x)
    print([item.size() for item in y])
    #print(net(x))
    #summary(net, (3, 224, 224))

if __name__ == '__main__':
    test()
