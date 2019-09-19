import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))

import torch
import torch.nn as nn

from modules.basic.pool import TopLeftPool, BottomRightPool, CenterPool
from modules.basic.basic import convolution2d, sigmoid
from loss.KeypointBasedLoss import focal_loss, ae_loss, regr_loss

def _transpose_and_gather_feat(feat, ind):
    # feat: bs * c * h * w,   ind: bs * max_tag_len
    feat = feat.permute(0, 2, 3, 1).contiguous() # bs * h * w * c
    feat = feat.view(feat.size(0), -1, feat.size(3))  # bs * (h * w) * c
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)  # c
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # bs * max_tag_len * c
    feat = feat.gather(1, ind)  # bs * max_tag_len * c
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(feat.size(0), -1, dim)
    return feat

class HeatmapHead(nn.Module):
    def __init__(self, inp_planes, num_classes):
        super(HeatmapHead, self).__init__()
        self.heat_pre = nn.Sequential(
            convolution2d(inp_planes, inp_planes, 3, stride=1, padding=1, bn=False),
            nn.Conv2d(inp_planes, num_classes, (1, 1))
        )
        self.focal_loss = focal_loss()
       
    def forward(self, feat, target=None):
        # feats: bs * c * h * w
        # targets:
        heat = self.heat_pre(feat)   # bs * num_classes * h * w
        heat = sigmoid(heat, 1e-4, 1-1e-4)
        if self.training:
            assert target is not None
            focal_loss = self.focal_loss(heat, target)
            return focal_loss
        else:
            return heat

class AssociateEmbeddingHead(nn.Module):
    def __init__(self, inp_planes, embedding_dim=1):
        super(AssociateEmbeddingHead, self).__init__()
        self.pre1 = nn.Sequential(
            convolution2d(inp_planes, inp_planes, 3, stride=1, padding=1, bn=False),
            nn.Conv2d(inp_planes, embedding_dim, (1, 1))
        )
        self.pre2 = nn.Sequential(
            convolution2d(inp_planes, inp_planes, 3, stride=1, padding=1, bn=False),
            nn.Conv2d(inp_planes, embedding_dim, (1, 1))
        )
        self.ae_loss = ae_loss()

    def forward(self, feat1, feat2, targets=None):
        # feat: bs * embedding_dim * h * w
        # targets: [bs * max_num, bs * max_num, bs * max_num]
        embedding1 = self.pre1(feat1)  # bs * embedding_dim * h * w
        embedding2 = self.pre2(feat2)
        if self.training:
            assert targets is not None
            [inds1, inds2, mask] = targets
            embedding1 = _transpose_and_gather_feat(embedding1, inds1)  # bs * max_tag_len * embedding_dim
            embedding2 = _transpose_and_gather_feat(embedding2, inds2)  # bs * max_tag_len * embedding_dim
            pull, push = self.ae_loss(embedding1, embedding2, mask)
            return pull, push
        else:
            return embedding1, embedding2

class RegressionHead(nn.Module):
    def __init__(self, inp_planes, dim=2):
        super(RegressionHead, self).__init__()
        self.pre = nn.Sequential(
            convolution2d(inp_planes, inp_planes, 3, stride=1, padding=1, bn=False),
            nn.Conv2d(inp_planes, dim, (1, 1))
        )
        self.regr_loss = regr_loss()

    def forward(self, feat, targets=None):
        # feat: bs * embedding_dim * h * w
        # targets: [bs * max_num, bs * max_num, bs * max_num]
        regr = self.pre(feat)  # bs * 2 * h * w
        if self.training:
            assert targets is not None
            [gt_regr, inds, mask] = targets
            regr = _transpose_and_gather_feat(regr, inds)  # bs * max_tag_len * 2
            regr_loss = self.regr_loss(regr, gt_regr, mask)
            return regr_loss
        else:
            return regr

class CornernetHead(nn.Module):
    def __init__(self, n, inp_planes, num_classes, embedding_dim=1, tl_heatmap_weight=1, br_heatmap_weight=1, \
                 pull_weight=1e-1, push_weight=1e-1, tl_regr_weight=1, br_regr_weight=1):
        super(CornernetHead, self).__init__()
        self.tl_heatmap_weight = tl_heatmap_weight
        self.br_heatmap_weight = br_heatmap_weight
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.tl_regr_weight = tl_regr_weight
        self.br_regr_weight = br_regr_weight
        self.tl_cnvs = nn.ModuleList([
            TopLeftPool(inp_planes) for _ in range(n)
        ])
        self.br_cnvs = nn.ModuleList([
            BottomRightPool(inp_planes) for _ in range(n)
        ])
        self.tl_heatmap_heads = nn.ModuleList([
            HeatmapHead(inp_planes, num_classes) for _ in range(n)
        ])
        self.br_heatmap_heads = nn.ModuleList([
            HeatmapHead(inp_planes, num_classes) for _ in range(n)
        ])
        self.ae_heads = nn.ModuleList([
            AssociateEmbeddingHead(inp_planes, 1) for _ in range(n)
        ])
        self.tl_regr_heads = nn.ModuleList([
            RegressionHead(inp_planes, 2) for _ in range(n)
        ])
        self.br_regr_heads = nn.ModuleList([
            RegressionHead(inp_planes, 2) for _ in range(n)
        ])

    def forward(self, feats, targets=None):
        if self.training:
            focal_loss, pull_loss, push_loss, regr_loss = 0., 0., 0., 0.
            for i, feat in enumerate(feats):
                gt_tl_heat, gt_br_heat, gt_tl_inds, gt_br_inds, gt_tl_regr, gt_br_regr, gt_mask = targets[i]
                tl_feat = self.tl_cnvs[i](feat)
                br_feat = self.br_cnvs[i](feat)
                focal_loss += self.tl_heatmap_heads[i](tl_feat, gt_tl_heat) * self.tl_heatmap_weight
                focal_loss += self.br_heatmap_heads[i](br_feat, gt_br_heat) * self.br_heatmap_weight
                pull_loss_, push_loss_ = self.ae_heads[i](tl_feat, br_feat, [gt_tl_inds, gt_br_inds, gt_mask])
                pull_loss += pull_loss_ * self.pull_weight
                push_loss += push_loss_ * self.push_weight
                regr_loss += self.tl_regr_heads[i](tl_feat, [gt_tl_regr, gt_tl_inds, gt_mask]) * self.tl_regr_weight
                regr_loss += self.br_regr_heads[i](br_feat, [gt_br_regr, gt_br_inds, gt_mask]) * self.br_regr_weight
            loss = focal_loss + pull_loss + push_loss + regr_loss
            return loss, focal_loss, pull_loss, push_loss, regr_loss
        else:
            tl_feat = self.tl_cnvs[i](feats[-1])
            br_feat = self.br_cnvs[i](feats[-1])
            tl_heat = self.tl_heatmap_heads[-1](tl_feat)
            br_heat = self.br_heatmap_heads[-1](br_feat)
            embedding1, embedding2 = self.ae_heads[-1](tl_feat, br_feat)
            tl_regr = self.tl_regr_heads[-1](tl_feat)
            br_regr = self.br_regr_heads[-1](br_feat)
            return (tl_heat, br_heat, embedding1, embedding2, tl_regr, br_regr)

class CenternetHead(nn.Module):
    def __init__(self, n, inp_planes, num_classes, embedding_dim=1, tl_heatmap_weight=1, br_heatmap_weight=1, ct_heatmap_weight=1, \
                 pull_weight=1e-1, push_weight=1e-1, tl_regr_weight=1, br_regr_weight=1, ct_regr_weight=1):
        super(CenternetHead, self).__init__()
        self.tl_heatmap_weight = tl_heatmap_weight
        self.br_heatmap_weight = br_heatmap_weight
        self.ct_heatmap_weight = ct_heatmap_weight
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.tl_regr_weight = tl_regr_weight
        self.br_regr_weight = br_regr_weight
        self.ct_regr_weight = ct_regr_weight
        self.tl_cnvs = nn.ModuleList([
            TopLeftPool(inp_planes) for _ in range(n)
        ])
        self.br_cnvs = nn.ModuleList([
            BottomRightPool(inp_planes) for _ in range(n)
        ])
        self.ct_cnvs = nn.ModuleList([
            CenterPool(inp_planes) for _ in range(n)
        ])
        self.tl_heatmap_heads = nn.ModuleList([
            HeatmapHead(inp_planes, num_classes) for _ in range(n)
        ])
        self.br_heatmap_heads = nn.ModuleList([
            HeatmapHead(inp_planes, num_classes) for _ in range(n)
        ])
        self.ct_heatmap_heads = nn.ModuleList([
            HeatmapHead(inp_planes, num_classes) for _ in range(n)
        ])
        self.ae_heads = nn.ModuleList([
            AssociateEmbeddingHead(inp_planes, 1) for _ in range(n)
        ])
        self.tl_regr_heads = nn.ModuleList([
            RegressionHead(inp_planes, 2) for _ in range(n)
        ])
        self.br_regr_heads = nn.ModuleList([
            RegressionHead(inp_planes, 2) for _ in range(n)
        ])
        self.ct_regr_heads = nn.ModuleList([
            RegressionHead(inp_planes, 2) for _ in range(n)
        ])

    def forward(self, feats, targets=None):
        if self.training:
            focal_loss, pull_loss, push_loss, regr_loss = 0., 0., 0., 0.
            for i, feat in enumerate(feats):
                gt_tl_heat, gt_br_heat, gt_ct_heat, gt_tl_inds, gt_br_inds, gt_ct_inds, gt_tl_regr, gt_br_regr, gt_ct_regr, gt_mask = targets[i]
                tl_feat = self.tl_cnvs[i](feat)
                br_feat = self.br_cnvs[i](feat)
                ct_feat = self.ct_cnvs[i](feat)
                focal_loss += self.tl_heatmap_heads[i](tl_feat, gt_tl_heat) * self.tl_heatmap_weight
                focal_loss += self.br_heatmap_heads[i](br_feat, gt_br_heat) * self.br_heatmap_weight
                focal_loss += self.ct_heatmap_heats[i](ct_feat, gt_ct_heat) * self.ct_heatmap_weight
                pull_loss_, push_loss_ = self.ae_heads[i](tl_feat, br_feat, [gt_tl_inds, gt_br_inds, gt_mask])
                pull_loss += pull_loss_ * self.pull_weight
                push_loss += push_loss_ * self.push_weight
                regr_loss += self.tl_regr_heads[i](tl_feat, [gt_tl_regr, gt_tl_inds, gt_mask]) * self.tl_regr_weight
                regr_loss += self.br_regr_heads[i](br_feat, [gt_br_regr, gt_br_inds, gt_mask]) * self.br_regr_weight
                regr_loss += self.ct_regr_heads[i](ct_feat, [gt_ct_regr, gt_ct_inds, gt_mask]) * self.ct_regr_weight
            loss = focal_loss + pull_loss + push_loss + regr_loss
            return loss, focal_loss, pull_loss, push_loss, regr_loss
        else:
            tl_feat = self.tl_cnvs[i](feats[-1])
            br_feat = self.br_cnvs[i](feats[-1])
            tl_heat = self.tl_heatmap_heads[-1](tl_feat)
            br_heat = self.br_heatmap_heads[-1](br_feat)
            embedding1, embedding2 = self.ae_heads[-1](tl_feat, br_feat)
            tl_regr = self.tl_regr_heads[-1](tl_feat)
            br_regr = self.br_regr_heads[-1](br_feat)
            return (tl_heat, br_heat, embedding1, embedding2, tl_regr, br_regr)


