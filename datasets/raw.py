from new.utils.common import printx, dir_check
from tqdm import  tqdm
import xml.etree.ElementTree as ET
import json
import random
import cv2
import numpy as np


import os

def get_bbxs_infos(str_file_xml, list_classes = None):
    tree = ET.parse(str_file_xml)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    list_bbxs = []
    list_clsids=[]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if list_classes is not None and cls not in list_classes or int(difficult) == 1: continue
        cls_id = list_classes.index(cls)
        xmlbox = obj.find('bndbox')
        bb = (max(1, float(xmlbox.find('xmin').text)),
              max(1, float(xmlbox.find('ymin').text)),
              min(w - 1, float(xmlbox.find('xmax').text)),
              min(h - 1, float(xmlbox.find('ymax').text)))
        list_bbxs.append(bb)
        list_clsids.append(cls_id)
    assert list_bbxs
    list_bbxs = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in list_bbxs]
    return (w,h), list_bbxs, list_clsids,

def construct_cache(str_dataset_name, str_dir_root, list_classes):
    assert str_dataset_name in ['VOC2007']
    dic_cache = {}
    if str_dataset_name == 'VOC2007':
        for str_mode in ['train', 'test', 'val']:
            dic_cache_mode = {}
            str_file_ids = os.path.join(str_dir_root, 'ImageSets/Main/{}.txt'.format(str_mode))
            list_ids = open(str_file_ids).read().split()
            for str_id in tqdm(list_ids):
                str_file_img = os.path.join(str_dir_root, 'JPEGImages/{}.jpg'.format(str_id))
                str_file_xml = os.path.join(str_dir_root, 'Annotations/{}.xml'.format(str_id))
                assert os.path.exists(str_file_img) and os.path.exists(str_file_xml)
                tu_size, list_bbxs, list_ids = get_bbxs_infos(str_file_xml, list_classes)
                dic_cache_mode['{}.jpg'.format(str_id)]={"list_bbxs":list_bbxs,
                                                         "list_clsids":list_ids,
                                                         'size':tu_size}
            dic_cache[str_mode] = dic_cache_mode
    return dic_cache

class MetaRawData(object):

    def __init__(self):
        self.dic_cache = None

    def load(self):
        pass

    def getitem(self, item, str_flag='train'):
        pass

class RawData(MetaRawData):

    def __init__(self, config):
        super(RawData, self).__init__()

        try:
            self.str_dir_root = config.str_dir_root
            self.str_dir_imgroot = os.path.join(self.str_dir_root, 'JPEGImages')
            self.list_classes = config.list_classes
            self.str_dir_cachefile = config.str_dir_cachefile
            assert dir_check(config.str_dir_root) and dir_check(self.str_dir_imgroot)
        except Exception as e:
            printx('err in RawData:{}'.format(e))
            raise e


        self.dic_cache = self.load()

    def load(self):
        if os.path.exists(self.str_dir_cachefile):
            with open(self.str_dir_cachefile, 'r') as fp:
                dic_cache = json.load(fp)
        else:
            with open(self.str_dir_cachefile, 'w') as fp:
                dic_cache = construct_cache(str_dataset_name='VOC2007',
                                            str_dir_root=self.str_dir_root,
                                            list_classes=self.list_classes)
                json.dump(dic_cache, fp, indent=4, ensure_ascii=True)
        return dic_cache

    def length(self, str_flag):
        assert str_flag in self.dic_cache
        return self.dic_cache[str_flag].__len__()

    def getitem(self, item, str_flag='train'):
        assert item in range(self.length(str_flag))
        dic_cache_flag = self.dic_cache[str_flag]
        str_img_id = list(dic_cache_flag.keys())[item]
        str_img_path = os.path.join(self.str_dir_imgroot, str_img_id)
        list2_bbxs = dic_cache_flag[str_img_id]['list_bbxs']
        list_clsids = dic_cache_flag[str_img_id]['list_clsids']
        tu_size = dic_cache_flag[str_img_id]['size']
        return str_img_path, np.array(list2_bbxs), np.array(list_clsids)[:,np.newaxis], tu_size,

    def cache_test(self):
        str_mode = random.choice(['train', 'test', 'val'])
        list_test_images = random.choices(list(self.dic_cache[str_mode].items()), k=10)
        for k, v in list_test_images:
            str_file_img = os.path.join(self.str_dir_imgroot, k)
            print(str_file_img, v)
            image = cv2.imread(str_file_img)
            cv2.namedWindow('result')
            cv2.imshow('result', image)
            cv2.waitKey()

if __name__ == '__main__':
    import new.config as config
    obj_raw = RawData(config=config)
    obj_raw.cache_test()
