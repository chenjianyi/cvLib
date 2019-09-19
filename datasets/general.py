import cv2
import numpy as np

from new.head.head import YOLOHEAD
from new.database.raw import RawData
from new.database import boxarg

import torch

class DATASET(object):

    def __init__(self, config, list_arguers=None, target_builder=None, str_flag='train'):
        self.obj_raw = RawData(config)
        self.list_arguers = list_arguers
        self.target_builder = target_builder
        self.str_flag = str_flag

    def __getitem__(self, item):
        # data loading
        str_img_path, np_bboxes, np_clsids, tu_size = self.obj_raw.getitem(item, str_flag=self.str_flag)
        np_img = cv2.imread(str_img_path)
        # boxes argument
        np_bboxes[:,2] += np_bboxes[:,0]
        np_bboxes[:,3] += np_bboxes[:,1]
        if self.list_arguers:
            tu_size = (448,448)
            for arguer in self.list_arguers:
                np_img, np_bboxes = arguer(np_img, np_bboxes)
        np_img, np_bboxes = np.ascontiguousarray(np_img), np.ascontiguousarray(np_bboxes)
        np_bboxes[:, 2] -= np_bboxes[:, 0]
        np_bboxes[:, 3] -= np_bboxes[:, 1]
        # boxes encoder
        np_bboxes_gt = np.concatenate([np_bboxes, np.ones_like(np_clsids), np_clsids], axis=1)
        ts_target = torch.tensor([])
        if self.target_builder is not None:
            ts_target = self.target_builder(tu_size=tu_size,  np_bboxes_gt=np_bboxes_gt)

        return str_img_path, tu_size, np_img, np_bboxes_gt, ts_target

    def _collate_fn(self, batch):
        list_str_imgs = [item[0] for item in batch]
        list_tu_sizes = [torch.tensor(item[1]) for item in batch]
        list_ts_imgs = [torch.tensor(item[2]) for item in batch]
        list_ts_bboxes_gt = [torch.tensor(item[3]) for item in batch]
        list_ts_targets = [item[4] for item in batch]
        return list_str_imgs, list_tu_sizes, list_ts_imgs, list_ts_bboxes_gt, list_ts_targets,

    def __len__(self):
        return self.obj_raw.length(str_flag=self.str_flag)

if __name__ == "__main__":
    import new.config as config
    from torch.utils.data import dataloader
    from tqdm import tqdm
    import time

    obj_head = YOLOHEAD(config)
    # obj_dataset = DATASET(config, list_arguers=boxarg.box_resize, encoder=obj_head.encoder)
    obj_dataset = DATASET(config,
                          list_arguers=[boxarg.box_resize,
                                        boxarg.box_random_horizon_flip,
                                        boxarg.box_random_vertical_flip],
                          target_builder=obj_head.encoder)
    obj_dataloader = dataloader.DataLoader(dataset= obj_dataset, batch_size=9, shuffle=True, num_workers=8,
                                           collate_fn=obj_dataset._collate_fn)
    # todo show some imgs through [DataArgument -> Encoder -> Decoder] cycle
    cv2.namedWindow('result')
    idx_batch = 0
    for (list_str_imgs, list_tu_sizes, list_ts_imgs, list_ts_bboxes_gt, list_ts_targets) in tqdm(obj_dataloader):
        for idx_img in range(len(list_str_imgs)):
            print('img: {}, size:{}'.format(list_str_imgs[idx_img], list_tu_sizes[idx_img]))
            img = list_ts_imgs[idx_img].numpy()
            np_boxes_gt = list_ts_bboxes_gt[idx_img].numpy()
            # np_boxes_gt = obj_head.decoder(tu_size=(448,448), ts_target = list_ts_targets[idx_img])
            np_bboxes = np_boxes_gt[:,:-2]
            for idx_box in range(np_bboxes.shape[0]):
                x, y, w, h = np_bboxes[idx_box]
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), thickness=3)
            cv2.imshow('result', img)
            cv2.waitKey()
            cv2.destroyWindow('result')
        idx_batch += 1
