import cv2
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import random

# https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html
def box_resize(np_img, np_bboxes, tu_size=(448,448)): # tu_size=(height, weight)
    list_bbx_bbs = [BoundingBox(*list_bbx) for list_bbx in np_bboxes.tolist()]
    ia_bbs = BoundingBoxesOnImage(list_bbx_bbs, shape=np_img.shape)
    image_rescaled = ia.imresize_single_image(np_img,tu_size)
    bbs_rescaled = ia_bbs.on(image_rescaled)
    list2_bbx_scaled = []
    for i in range(len(bbs_rescaled.bounding_boxes)):
        cur_bbx = bbs_rescaled.bounding_boxes[i]
        x1, y1, x2, y2 = int(cur_bbx.x1), int(cur_bbx.y1), int(cur_bbx.x2), int(cur_bbx.y2)
        list2_bbx_scaled.append([x1,y1,x2,y2])
    return image_rescaled, np.array(list2_bbx_scaled)

def box_random_horizon_flip(np_img, np_bboxes, p=0.5):
    np_img_a, np_bbxes_a = np_img.copy(), np_bboxes.copy()

    np_img_center = np.array(np_img_a.shape[:2])[::-1] // 2
    np_img_center = np.hstack((np_img_center, np_img_center))
    if random.random() < p:
        np_img_a = np_img_a[:, ::-1, :]
        np_bbxes_a[:, [0, 2]] += 2 * (np_img_center[[0, 2]] - np_bbxes_a[:, [0, 2]])
        box_w = abs(np_bbxes_a[:, 0] - np_bbxes_a[:, 2])
        np_bbxes_a[:, 0] -= box_w
        np_bbxes_a[:, 2] += box_w
    return np.ascontiguousarray(np_img_a), np_bbxes_a.astype(int)

def box_random_vertical_flip(np_img, np_bboxes, p=0.5):
    np_img_a, np_bbxes_a = np_img.copy(), np_bboxes.copy()

    np_img_center = np.array(np_img_a.shape[:2])[::-1] // 2
    np_img_center = np.hstack((np_img_center, np_img_center))
    if random.random() < p:
        np_img_a = np_img_a[::-1,:,:]
        np_bbxes_a[:, [1, 3]] += 2 * (np_img_center[[1, 3]] - np_bbxes_a[:, [1, 3]])
        box_w = abs(np_bbxes_a[:, 1] - np_bbxes_a[:, 3])
        np_bbxes_a[:, 1] -= box_w
        np_bbxes_a[:, 3] += box_w
    return np.ascontiguousarray(np_img_a), np_bbxes_a.astype(int)


if __name__ == '__main__':
    img = cv2.imread('/home/gsliu/data/share/dataset/VOC2007/JPEGImages/002102.jpg')
    np_img = img
    np_bbxes = np.array([[100, 80, 400, 180]])
    # todo visualization
    # for idx in range(np_bbxes.shape[0]):
    #     x1, y1, x2, y2 = np_bbxes[idx]
    #     cv2.rectangle(np_img, (x1, y1), (x2, y2), color=(0, 255, 0))
    # cv2.imshow('before', np_img)

    np_img_a, np_bbxes_a = box_random_horizon_flip(np_img, np_bbxes)
    # np_img_a, np_bbxes_a = box_resize(np_img, np_bbxes, 448)
    for idx in range(np_bbxes_a.shape[0]):
        x1, y1, x2, y2  = np_bbxes_a[idx]
        cv2.rectangle(np_img_a, (x1, y1), (x2, y2), color=(255, 255, 0))
    cv2.imshow('after', np_img_a)
    cv2.waitKey()

    cv2.destroyAllWindows()
