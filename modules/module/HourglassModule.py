import torch
import torch.nn as nn
from basic import convolution2d, residual

def make_layer(k, inp_dim, out_dim, modules, layer=convolution2d):
    layers = [layer(inp_dim, out_dim, k)]
    for _ in range(1, modules):
        layers.append(layer(out_dim, out_dim, k))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution2d):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(inp_dim, inp_dim, k))
    layers.append(layer(inp_dim, out_dim, k))
    return nn.Sequential(*layers)

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_merge_layer(dim):
    return MergeUp()

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

class HourglassModule(nn.Module):
    """Hourglass Module
    Args:
        n:
        dims:
        modules:
    """
    def __init__(
        self, n, dims, modules, layer, 
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer
    ):
        super(HourglassModule, self).__init__()
        self.n = n  # n = 3
        curr_mod = modules[0]
        next_mod = modules[1]
        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1 = make_up_layer(3, curr_dim, curr_dim, curr_mod, layer)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(3, curr_dim, next_dim, curr_mod, layer)

        self.up2 = make_unpool_layer(curr_dim)
        self.low2 = HourglassModule(
            n - 1, dims[1: ], modules[1: ], layer,
            make_up_layer=make_up_layer, make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer, make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer
        ) if self.n > 1 else \
        make_low_layer(3, next_dim, next_dim, next_mod, layer=layer)

        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod, layer=layer
        )

        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        # print('x', x.size())
        # x: bs * curr_dim * h * w
        up1 = self.up1(x)  # bs * curr_dim * h * w, [convolution] * curr_mod
        max1 = self.max1(x) # bs * curr_dim * (h) * (w), nn.Sequential
        low1 = self.low1(max1)  # bs * next_dim * (h // 2) * (w // 2)
        if self.n > 1:
            low2, mergs = self.low2(low1)  # bs * next_dim * (h // 2) * (w // 2)
        else:
            low2, mergs = self.low2(low1), []
        low3 = self.low3(low2)  # bs * curr_dim * (h // 2) * (w // 2)
        up2 = self.up2(low3)  # bs * curr_dim * h * w
        merg = self.merg(up1, up2)  # bs * curr_dim * h * w
        mergs.append(merg)
        return merg, mergs


