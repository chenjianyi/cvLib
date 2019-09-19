import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn
import torch.nn.functional as F

def log_sum_exp(x):
    # x: (bs * num_priors) * num_classes
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max  # (bs * num_priors) * 1

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    """

    def __init__(self, **kwargs):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = kwargs.get("num_classes")
        self.threshold = kwargs.get("overlap_thresh")
        self.background_label = kwargs.get("bkg_label") if kwargs.get("bkg_label") else 0
        self.do_neg_mining = kwargs.get("do_neg_mining") if kwargs.get("do_neg_mining") else True
        self.negpos_ratio = kwargs.get("negpos_ratio")

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions: tuple or list, [bs * num_priors * 4, bs * num_priors * num_classes]
            targets: bs * num_objs * 5
        """
        loc_data, conf_data = predictions
        loc_t, conf_t = targets
        bs = loc_data.size(0)
        num_priors = (predictions[0].size(1))
        num_classes = self.num_classes

        pos = conf_t != self.background_label
        neg = conf_t == self.background_label

        # Localization Loss (Smooth L1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # bs * num_priors * 4
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        if self.do_neg_mining:
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # (bs * num_priors) * 1
            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0
            loss_c = loss_c.view(bs, -1)  # bs * num_priors
            _, loss_idx = loss_c.sort(1, descending=True)  # bs * num_priors
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)  # bs * 1
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1, min=1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)  # bs * num_priors * num_classes
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)  # bs * num_priors * num_classes
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
