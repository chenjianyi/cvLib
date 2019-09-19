import torch
import torch.nn as nn

class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss, self).__init__()

    def forward(self, pred, gt):
        # preds: bs * out_dim * h * w,  gt: bs * out_dim * output_size[0] * output_size[1]
        pos_inds = gt.eq(1)  # bs * out_dim * output_size[0] * output_size[1]
        neg_inds = gt.lt(1)  # bs * out_dim * output_size[0] * output_size[1]
        neg_weights = torch.pow(1 - gt[neg_inds], 4)  # bs * out_dim * output_size[0] * output_size[1]
    
        loss = 0
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights.cuda()
    
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum() * 5
        neg_loss = neg_loss.sum()
    
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    
        return loss

class ae_loss(nn.Module):
    def __init__(self):
        super(ae_loss, self).__init__()

    def forward(self, tag0, tag1, mask):
        # tag0: bs * max_tag_len * 1, tag1: bs * max_tag_len * 1
        # mask: bs * max_tag_len
        num = mask.sum(dim=1, keepdim=True).float()  # bs * 1
        tag0 = tag0.squeeze(-1)  # bs * max_tag_len
        tag1 = tag1.squeeze(-1)  # bs * max_tag_len
        tag_mean = (tag0 + tag1) / 2
    
        tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4).cuda()
        tag0 = tag0[mask].sum()
        tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4).cuda()
        tag1 = tag1[mask].sum()
        pull = tag0 + tag1
    
        mask = mask.unsqueeze(1) + mask.unsqueeze(2)  # bs * max_tag_len * max_tag_len
        mask = mask.eq(2)  # bs * max_tag_len * max_tag_len
        num = num.unsqueeze(2)  # bs * 1 * 1
        num2 = (num - 1) * num  # bs * 1 * 1
        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)  # bs * max_tag_len * max_tag_len
        dist = 1 - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - 1 / (num + 1e-4).cuda()
        dist = dist / (num2 + 1e-4).cuda()
        dist = dist[mask]
        push = dist.sum()
        return pull, push

class regr_loss(nn.Module):
    def __init__(self):
        super(regr_loss, self).__init__()

    def forward(self, regr, gt_regr, mask):
        # regr: bs * max_tag_len * 2, gt_regr: bs * max_tag_len * 2, mask: bs * max_tag_len
        num = mask.float().sum()  # (1, )
        mask = mask.unsqueeze(2).expand_as(gt_regr)  # bs * max_tag_len * 2
    
        regr = regr[mask]  #
        gt_regr = gt_regr[mask]
    
        regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr.cuda(), reduction='sum')
        regr_loss = regr_loss / (num + 1e-4).cuda()
        return regr_loss
