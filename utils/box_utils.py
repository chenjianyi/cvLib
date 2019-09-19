import torch
import torch.nn  as nn

import numpy as np

def point_form(boxes):
    """Convert (xctr, yctr, w, h) to (xmin, ymin, xmax, ymax)
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2, boxes[:, :2] + boxes[:, 2:]/2), 1)

def center_size(boxes):
    """Convert (xmin, ymin, xmax, ymax) to (xctr, yctr, w, h)
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2, boxes[:, 2:] - boxes[:, :2], 1)

def intersect(box_a, box_b):
    # box_a: A * 4
    # box_b: B * 4
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), \
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), \
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)  # A * B * 2
    return inter[:, :, 0] * inter[:, :, 1]  # A * B

def jaccard(box_a, box_b):
    # box_a: A * 4
    # box_b: B * 4
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter) # A * B
    area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter) # A * B
    union = area_a + area_b - inter
    return inter / union  # A * B

