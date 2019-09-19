import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import product as product
import math

from loss import MultiBoxLoss

from utils.box_utils import jaccard, point_form

def match(threshold, truths, priors, variances, labels, bkg_label):
    overlaps = jaccard(truths, point_form(priors))  # num_objs * num_priors
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)  # (num_priors, )
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)  # (num_objs, )
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # num_priors * 4
    conf = labels[best_truth_idx]  # (num_priors, )
    conf[best_truth_overlap < threshold] = bkg_label  # background
    loc = encode(matches, priors, variances)  # num_priors * 4
    return conf, loc

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # num_priors * 4

class PriorsBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    """
    def __init__(self, **kwargs):
        super(PriorsBox, self).__init__()
        self.feature_maps = kwargs.get("feature_maps")
        self.image_size = kwargs.get("min_dim")
        self.aspect_ratios = kwargs.get("aspect_ratios")
        self.steps = kwargs.get("steps")
        self.min_sizes = kwargs.get("min_sizes")
        self.max_sizes = kwargs.get("max_sizes")
        self.clip = kwargs.get("clip")
        self.num_priors = len(self.aspect_ratios)

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*math.sqrt(ar), s_k/math.sqrt(ar)]
                    mean += [cx, cy, s_k/math.sqrt(ar), s_k*math.sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class Targets_builder():
    def __init__(self, priors, **loss_setting):
        self.priors = priors
        self.num_classes = loss_setting.get("num_classes")
        self.threshold = loss_setting.get("overlap_thresh")
        self.variances = loss_setting.get('variances') if loss_setting.get('variances') else [0.1, 0.2]
        self.background_label = loss_setting.get("bkg_label") if loss_setting.get("bkg_label") else 0

    def forward_batch(self, targets):
        # targets: list, [n1 *5, n2 * 5] 
        bs = len(targets)
        num_priors = (self.priors.size(0))
        num_classes = self.num_classes

        loc_t = torch.Tensor(bs, num_priors, 4).cuda()
        conf_t = torch.LongTensor(bs, num_priors).cuda()
        for idx in range(bs):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = self.priors.data
            conf, loc = match(self.threshold, truths, defaults, self.variances, labels, self.background_label)
            loc_t[idx] = loc
            conf_t[idx] = conf
        return loc_t, conf_t

    def forward_single(self, target):
        # target: torch.tensor, n * 5
        targets = [target]
        loc_t, conf_t = self.forward_batch(targets)
        loc_t, conf_t = loc_t[0], conf_t[0]
        return loc_t, conf_t

class SSDHead(nn.Module):
    def __init__(self, inplanes, num_anchors, num_scales, num_classes, loss_setting, priors_setting):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes
        loc_ = list()
        conf_ = list()
        if isinstance(inplanes, int):
            inplanes = [inplanes] * num_scales
        for i in range(num_scales):
            loc_.append(nn.Conv2d(inplanes[i], 4 * num_anchors, kernel_size=3, stride=1, padding=1))
            conf_.append(nn.Conv2d(inplanes[i], num_classes*6, kernel_size=3, stride=1, padding=1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        self.loss_setting = loss_setting
        self.Loss = MultiBoxLoss(**loss_setting)

        self.priors_setting = priors_setting
        self.variances = loss_setting.get('variances') if loss_setting.get('variances') else [0.1, 0.2]
        self.priors = self._build_priors()

    def _build_priors(self):
        Priors = PriorsBox(**self.priors_setting)
        priors = Priors.forward().cuda()
        return priors

    def targets_build(self, targets):
        targets_builder = Targets_builder(self.priors, **self.loss_setting)
        return targets_builder.forward_batch(targets)

    def loss(self):
        return self.Loss

    def forward(self, feats, targets=None):
        loc = []
        conf = []
        for (x, l, c) in zip(feats, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.training:
            outputs = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            built_targets = self.targets_build(targets)
            loss_l, loss_c = self.Loss(outputs, built_targets)
            loss = loss_l + loss_c
            return loss, loss_l, loss_c
        else:
            predictions = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            bboxes = self.decode(predicions)
            return bboxes

    def decode(self, predictions):
        loc, conf = predictions
        conf = F.softmax(conf, dim=2)
        loc_data, conf_data = loc.data, conf.data
        priors_data = self.priors.data
        bs = loc_data.size(0)
        num_priors = priors_data.size(0)
        boxes = torch.zeros(bs, num_priors, 4)
        scores = torch.zeros(bs, num_priors, self.num_classes)
        if loc_data.is_cuda:
            boxes = boxes.cuda()
            scores = scores.cuda()
        if bs == 1:
            conf_preds = conf_data.unsqueeze(0)
        else:
            conf_preds = conf_data.view(bs, num_priors, self.num_classes)

        # Decode predictions into bboxes
        for i in range(bs):
            decoded_boxes = self._decode(loc_data[i], priors_data, self.variances)
            conf_scores = conf_preds[i].clone()
            boxes[i] = decoded_boxes
            scores[i] = conf_scores
        bboxes = torch.cat([boxes, scores], -1)
        return bboxes

    def _decode(self, loc, priors, variances):
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2: ] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes  # bs * num_priors * 4, 4 ->(x1, y1, x2, y2)

def test():
    conf = {
        'feature_maps': [40, 20, 10, 5, 3, 1],
        'min_dim': 320,
        'steps': [8, 16, 32, 64, 107, 320],
        'min_sizes': [25.6, 48.0, 105.6, 163.2, 220.8, 278.4],
        'max_sizes': [48.0, 105.6, 163.2, 220.8, 278.4, 336.0],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        'variance': [0.1, 0.2],
        'clip': True
    }

    a = PriorsBox(**conf)
    output = a.forward()
    print(output)
    print(output.size())

if __name__ == '__main__':
    test()
