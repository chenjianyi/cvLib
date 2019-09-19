import math
import torch
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.002)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    #pred: bs * (13 * 13 + 26 * 26 + 52 * 52) * (5 + num_classes), 5 -> [x, y, w, h, conf]
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]  # n * (5 + num_classes)
        if image_pred.size(0) == 0:
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5: (5 + num_classes)], 1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)  # n * 7, 7->(x, y, w, h, conf, class_conf, class_pred)
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]  # m * 7
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]  # m * 7
            #Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                max_detections.append(detections_class[0].unsqueeze(0))  # (7, )
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            max_detections = torch.cat(max_detections).data
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    return output

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1: ] != mrec[: -1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class TargetBuilder():
    def __init__(self, anchors, num_classes, dims=[13, 26, 52], ignore_thres=0.5, img_size=416):
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_bbox_attrs = 5 + num_classes
        self.ignore_thres = ignore_thres
        self.dims = dims
        self.img_size = img_size
        if len(dims) == 3:
            self.mask = [[6 ,7, 8], [3, 4, 5], [0, 1, 2]]
        elif len(dims) == 2:
            self.mask = [[3, 4, 5], [0, 1, 2]]

    def build(self, target, dim, mask_i):
        anchors = [self.anchors[i] for i in mask_i]
        nA = len(anchors)
        nC = self.num_classes
        mask = np.zeros((nA, dim, dim), 'float32')
        conf_mask = np.ones((nA, dim, dim), 'float32')
        #tx, ty, tw, th, tconf, tcls are the targets to be predicted
        tx = np.zeros((nA, dim, dim), 'float32')
        ty = np.zeros((nA, dim, dim), 'float32')
        tw = np.zeros((nA, dim, dim), 'float32')
        th = np.zeros((nA, dim, dim), 'float32')
        tconf = np.zeros((nA, dim, dim), 'float32')
        tcls = np.zeros((nA, dim, dim, nC), 'float32')

        stride = self.img_size / dim
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]
        for t in range(target.shape[0]):
            if target[t].sum() <= 1e-5:
                continue
            gt_x = target[t, 1] * dim
            gt_y = target[t, 2] * dim
            gt_w = target[t, 3] * dim
            gt_h = target[t, 4] * dim
            #(gi, gj): which grid the box centre lie in
            gt_i = int(gt_x)
            gt_j = int(gt_y)
            gt_box = np.array([[0, 0, gt_w, gt_h]], dtype='float32') # 1 * 4
            anchor_shapes = np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1)
            anchor_shapes = anchor_shapes.astype('float32')
            #anchor_shapes: len(scaled_anchors) * 4: [ [0, 0, scaled_aw, scaled_ah], ... ]
            anchor_ious = bbox_iou_np(gt_box, anchor_shapes) #shape: (len(anchors), )
            conf_mask[anchor_ious > self.ignore_thres] = 0
            best_n = np.argmax(anchor_ious)
            mask[best_n, gt_j, gt_i] = 1
            conf_mask[best_n, gt_j, gt_i] = 1
            #build tx, ty, tw, th
            tx[best_n, gt_j, gt_i] = gt_x - gt_i
            ty[best_n, gt_j, gt_i] = gt_y - gt_j
            tw[best_n, gt_j, gt_i] = math.log(gt_w / scaled_anchors[best_n][0] + 1e-16)
            th[best_n, gt_j, gt_i] = math.log(gt_h / scaled_anchors[best_n][1] + 1e-16)
            tcls[best_n, gt_j, gt_i, int(target[t, 0])] = 1
            tconf[best_n, gt_j, gt_i] = 1
        #print(np.sum(conf_mask == 1), np.sum(conf_mask * tconf == 1))
        meta = np.stack([mask, conf_mask, tx, ty, tw, th, tconf], -1) #shape: nA * dim * dim * 7
        out = np.concatenate([meta, tcls], -1) #shape: nA * dim *dim * (7 + nC)
        return out

    def forward(self, target):
        outs = []
        for i in range(len(self.dims)):
            outs.append(self.build(target, self.dims[i], self.mask[i]))
        return outs

class TargetBuilder2():
    # create at 2018.11.12, within the quality of bbox in consideration
    def __init__(self, anchors, num_classes, dims=[13, 26, 52], ignore_thres=0.5, img_size=416):
        self.anchors = anchors
        self.num_classes = num_classes
        self.num_bbox_attrs = 5 + num_classes
        self.ignore_thres = ignore_thres
        self.dims = dims
        self.img_size = img_size
        if len(dims) == 3:
            self.mask = [[6 ,7, 8], [3, 4, 5], [0, 1, 2]]
        elif len(dims) == 2:
            self.mask = [[3, 4, 5], [0, 1, 2]]

    def build(self, target, dim, i):
        anchors = [anchor for anchor in self.anchors[i]]
        nA = len(anchors)
        nC = self.num_classes
        mask = np.zeros((nA, dim, dim), 'float32')
        conf_mask = np.ones((nA, dim, dim), 'float32')
        #tx, ty, tw, th, tconf, tcls are the targets to be predicted
        tx = np.zeros((nA, dim, dim), 'float32')
        ty = np.zeros((nA, dim, dim), 'float32')
        tw = np.zeros((nA, dim, dim), 'float32')
        th = np.zeros((nA, dim, dim), 'float32')
        tconf = np.zeros((nA, dim, dim), 'float32')
        tcls = np.zeros((nA, dim, dim, nC), 'float32')

        stride = self.img_size / dim
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]
        for t in range(target.shape[0]):
            if target[t].sum() <= 1e-5:
                continue
            gt_x = target[t, 2] * dim
            gt_y = target[t, 3] * dim
            gt_w = target[t, 4] * dim
            gt_h = target[t, 5] * dim
            #(gi, gj): which grid the box centre lie in
            gt_i = int(gt_x)
            gt_j = int(gt_y)
            gt_box = np.array([[0, 0, gt_w, gt_h]], dtype='float32') # 1 * 4
            anchor_shapes = np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1)
            anchor_shapes = anchor_shapes.astype('float32')
            #anchor_shapes: len(scaled_anchors) * 4: [ [0, 0, scaled_aw, scaled_ah], ... ]
            anchor_ious = bbox_iou_np(gt_box, anchor_shapes) #shape: (len(anchors), )
            #if (int(target[t, 1]) == 1):
            #    conf_mask[anchor_ious >= self.ignore_thres] = 0
            if (int(target[t, 1]) == 1):
                conf_mask[anchor_ious >= self.ignore_thres] = 0
                best_n = np.argmax(anchor_ious)
                mask[best_n, gt_j, gt_i] = 1
                conf_mask[best_n, gt_j, gt_i] = 1

                #build tx, ty, tw, th
                tx[best_n, gt_j, gt_i] = gt_x - gt_i
                ty[best_n, gt_j, gt_i] = gt_y - gt_j
                tw[best_n, gt_j, gt_i] = math.log(gt_w / scaled_anchors[best_n][0] + 1e-16)
                th[best_n, gt_j, gt_i] = math.log(gt_h / scaled_anchors[best_n][1] + 1e-16)
                tcls[best_n, gt_j, gt_i, int(target[t, 0])] = 1
                tconf[best_n, gt_j, gt_i] = 1
                #tconf[anchor_ious > self.ignore_thres] = 1
        #print(conf_mask.shape, np.sum(conf_mask == 1), np.sum(conf_mask * tconf == 1), np.sum(tconf == 1))
        meta = np.stack([mask, conf_mask, tx, ty, tw, th, tconf], -1) #shape: nA * dim * dim * 7
        out = np.concatenate([meta, tcls], -1) #shape: nA * dim *dim * (7 + nC)
        return out

    def forward(self, target):
        outs = []
        for i in range(len(self.dims)):
            outs.append(self.build(target, self.dims[i], i))
        return outs

def bbox_iou_np(box1, box2, x1y1x2y2=True):
    #x1y1x2y2 indicates whether is left_top and right_down points
    if not x1y1x2y2:  #is centre point
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    #caculate the left_top and right_down points of intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    #intersection rectangle area and union area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1 + 1, 0.0) *\
                 np.maximum(inter_rect_y2 - inter_rect_y1 + 1, 0.0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area + 1e-16
    #iou
    iou = inter_area / union_area
    return iou

def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

if __name__ == '__main__':
    recall = np.array([1./6, 1./6, 1./6, 1./3, 1./3, 1./3])
    precision = np.array([0.25, 0.25, 0.25, 0.5, 0.5, 0.5])
    ap = compute_ap(recall, precision)
    print(ap)
