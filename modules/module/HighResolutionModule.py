import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from block import BasicBlock, Bottleneck
from basic import convolution2d

class HighResolutionModule(nn.Module):
    """High Resolution Module
    Arguments:
        `num_branches`:
        `blocks`:
        `num_blocks`:
        `num_inchannels`:
        `num_channels`:
        `fuse_method`:
        `multi_scale_output`:
    Outputs:
    """
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, \
                 fuse_method, multi_scale_output=True, bias=False, sn=False, bn=True, act_fun='relu'):
        super(HighResolutionModule, self).__init__()
        self.bias, self.sn, self.bn, self.act_fun = bias, sn, bn, act_fun
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.muti_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        """ Check the params setting is legal or not!
            assert num_branches == len(num_blocks) == len(num_channels) == len(num_in_channels)
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            print(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            print(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            inp_planes = self.num_inchannels[branch_index]
            out_planes = num_channels[brach_index] * lock.expansion
            downsample = convolution2d(inp_planes, out_planes, kernel_size=1, stride=stride, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun)

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.muti_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        convolution2d(num_inchannels[j], num_inchannels[i], kernel_size=1, stride=1, padding=0, bias=self.bias, sn=self.sn, bn=self.bn, act_fun='none'),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest'),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                convolution2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=self.bias, sn=self.sn, bn=self.bn, act_fun='none')
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                convolution2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=self.bias, sn=self.sn, bn=self.bn, act_fun=self.act_fun)
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse
