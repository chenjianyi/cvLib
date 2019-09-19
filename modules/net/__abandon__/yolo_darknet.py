import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable

def parse_model_config(conf_path):
    with open(conf_path, 'r') as conf_file:
        lines = conf_file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):
            if line[1: -1].rstrip().lstrip() == 'yolo':
                continue
            module_defs.append({})
            module_defs[-1]['type'] = line[1: -1].rstrip().lstrip()
            if 'convolutional' in module_defs[-1]['type']:
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            module_defs[-1][key.strip()] = value.strip()
    return module_defs

def create_modules(module_defs):
    output_filters = [3]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if 'convolutional_before1_yolo' == module_def['type']:
            modules.add_module('empty_%d' % i, nn.Sequential())  
        elif 'convolutional' in module_def['type']:
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],\
                out_channels=filters, kernel_size=kernel_size, stride=int(module_def['stride']),\
                padding=pad, bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'upsample':
            upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            #upsample = nn.functional.interpolate(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            modules.add_module('empty_%d' % i, EmptyLayer())
        module_list.append(modules)
        output_filters.append(filters)
    return module_list

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class Darknet(nn.Module):
    def __init__(self, config_path='./yolo_darknet53.cfg'):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.module_list = create_modules(self.module_defs)

    def forward(self, x, targets=None):
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        outputs = []
        yolo_num = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'convolutional_before2_yolo', 'convolutional_before1_yolo']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(i) for i in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            layer_outputs.append(x)
            if module_def['type'] == 'convolutional_before2_yolo':
                outputs.append(x)

        outputs = outputs[::-1]
        return outputs

def test():
    from tools import summary
    net = Darknet('yolo_darknet53.cfg').cuda()
    x = torch.Tensor(1, 3, 416, 416).uniform_().cuda()
    y = net(x)
    print(len(y), y[0].size(), y[1].size(), y[2].size())
    summary(net, (3, 416, 416))

if __name__ == '__main__':
    test()
