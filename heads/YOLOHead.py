import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class YOLOv3Head(nn.Module):
    def __init__(self, anchors, num_classes, img_dim, g_dim, n_features, lambda_coord=0.2, eps=1e-8):
        super(YOLOv3Head, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim
        self.g_dim = g_dim
        self.lambda_coord = lambda_coord
        self.eps = eps
        self.grid_x, self.gird_y = self._make_grid(g_dim)
        self.anchor_w, self.anchor_h = self._make_anchor(g_dim)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        out_dim = (self.num_classes + 5) * len(self.anchors)
        if n_features != out_dim:
            self.conv_before_yolo = nn.Conv2d(n_features, out_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_before_yolo = nn.Sequential()

    def _make_grid(self, g_dim):
        x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim, 1).unsqueeze(0).unsqueeze(0)  # 1 * 1 * g_dim * g_dim
        x = x.repeat(1, self.num_anchors, 1, 1)  # 1 * 3 * g_dim * g_dim
        y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim, 1).t().unsqueeze(0).unsqueeze(0)
        y = y.repeat(1, self.num_anchors, 1, 1)
        return x.type(torch.cuda.FloatTensor), y.type(torch.cuda.FloatTensor)  # 1 * 3 * g_dim * g_dim

    def _make_anchor(self, g_dim):
        stride = self.img_dim / g_dim
        scaled_anchors = [(anchor_w / stride, anchor_h / stride) for anchor_w, anchor_h in self.anchors]  # 3 * 2
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))  # 3 * 1
        anchor_w = anchor_w.unsqueeze(-1).repeat(1, g_dim, g_dim)  # 3 * 1 * 1 -> 3 * g_dim * g_dim
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_h = anchor_h.unsqueeze(-1).repeat(1, g_dim, g_dim)
        return anchor_w.unsqueeze(0), anchor_h.unsqueeze(0)  #[1 * 3 * g_dim * g_dim]

    def forward(self, x, targets=None):
        # x: bs * c * h * w
        x = self.conv_before_yolo(x)
        bs, c, h, w = x.size()
        stride = self.img_dim / self.g_dim
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        prediction = x.view(bs, self.num_anchors, self.bbox_attrs, self.g_dim, self.g_dim).permute(0, 1, 3, 4, 2).contiguous()
        # prediction: bs * 3 * g_dim * g_dim * (5 + num_classes)

        if self.training:
            assert targets is not None  # targets: num_anchors * g_dim * g_dim * (7 + num_classes)
            return self.loss(prediction, targets)
        else:
            outputs = self.decode(predictions)
            return outputs

    def loss(self, prediction, targets):
        # targets
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        mask = targets[..., 0].type(FloatTensor)
        conf_mask = targets[..., 1].type(FloatTensor)
        tx = targets[..., 2].type(FloatTensor)
        ty = targets[..., 3].type(FloatTensor)
        tw = targets[..., 4].type(FloatTensor)
        th = targets[..., 5].type(FloatTensor)
        tconf = targets[..., 6].type(FloatTensor)
        tcls = targets[..., 7: ].type(FloatTensor)
        cls_mask = mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_classes).type(FloatTensor)  # 1 * num_anchors * g_dim * g_dim * num_classes

        # predictions
        eps = self.eps
        x = torch.sigmoid(prediction[..., 0]).clamp(eps, 1-eps)  # bs * 3 * g_dim * g_dim
        y = torch.sigmoid(prediction[..., 1]).clamp(eps, 1-eps)  # bs * 3 * g_dim * g_dim
        w = prediction[..., 2]  # bs * 3 * g_dim * g_dim
        h = prediction[..., 3]  # bs * 3 * g_dim * g_dim
        conf = torch.sigmoid(prediction[..., 4]).clamp(eps, 1-eps)  # bs * 3 * g_dim * g_dim
        pred_cls = torch.sigmoid(prediction[..., 5: ]).clamp(eps, 1-eps)  # bs * 3 * g_dim * g_dim * num_classes
 
        # loss
        xywh_factor = x.nelement() / (mask.sum() + 1.0)
        conf_factor = conf.nelement() / (conf_mask.sum() + 1.0)
        cls_factor = pred_cls.nelement() / (cls_mask.sum() + 1.0)
        loss_xy = self.lambda_coord * (self.mse_loss(x * mask, tx * mask) + self.mse_loss(y * mask, ty * mask)) * xywh_factor
        loss_wh = self.lambda_coord * (self.mse_loss(w * mask, tw * mask) + self.mse_loss(h * mask, th * mask)) * xywh_factor
        loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask) * conf_factor
        loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask) * cls_factor
        loss = loss_xy + loss_wh + loss_conf + loss_cls
        nShots = (conf >= 0.2).sum().item()

        return loss, loss_xy.item(), loss_wh.item(), loss_conf.item(), loss_cls.item(), nShots

    def decode(self, prediction):
        pred_boxes = FloatTensor(prediction[..., : 4].shape)  # bs * 3 * g_dim * g_dim * 4
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.gird_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        output = torch.cat([pred_boxes.view(bs, -1, 4) * stride, conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)], -1)
        return output.data  # bs * (3*g_dim*g_dim) * (5+num_classes)
