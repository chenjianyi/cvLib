from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import pickle, os, cv2
import torch
from multiprocessing import Pool
import glob
from PIL import Image

class ObjDataset(Dataset):
    def __init__(self, data, target_builder, img_size=416, transform=None, transform_path=None):
        with open(data, 'rb') as f:
            self.files = pickle.load(f)
        self.imgs = []
        for key in self.files.keys():
            self.imgs.append(key)
        self.N = len(self.files)
        self.img_size = img_size
        self.target_builder = target_builder
        self.transform = transform
        self.transform_path = transform_path

    def __getitem__(self, index):
        img_key = self.imgs[index % self.N]
        if self.transform_path is None:
            img_path = img_key
        else:
            img_path = self.transform_path(img_key)
        meta = self.files[img_key]
        try:
            img = cv2.imread(img_path)
            h, w, c = img.shape
            dim_diff = np.abs(h - w)
            pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
            pad = ( (pad1, pad2), (0, 0), (0, 0) ) if h <= w else ( (0, 0), (pad1, pad2), (0, 0)  )
            input_img = np.pad(img, pad, 'constant', constant_values=128)
            padded_h, padded_w, _ = input_img.shape
            input_img = cv2.resize(input_img, (self.img_size, self.img_size))
            if input_img.shape[-1] != 3:
                input_img = input_img[:, :, :3]
            if len(meta) > 0:
                labels = np.array(meta)
                print(labels)
                x1 = w * (labels[:, 1] - labels[:, 3] / 2)
                y1 = h * (labels[:, 2] - labels[:, 4] / 2)
                x2 = w * (labels[:, 1] + labels[:, 3] / 2)
                y2 = h * (labels[:, 2] + labels[:, 4] / 2)
                x1 += pad[1][0]
                y1 += pad[0][0]
                x2 += pad[1][0]
                y2 += pad[0][0]
                labels[:, 1] = ((x1 + x2) / 2) / padded_w
                labels[:, 2] = ((y1 + y2) / 2) / padded_h
                labels[:, 3] *= w / padded_w
                labels[:, 4] *= h / padded_h
            else:
                labels = np.zeros((2, 5), 'float32')
        except:
            print(img_path)
            input_img = 127 * np.ones((self.img_size, self.img_size, 3), 'uint8')
            labels = np.zeros((2, 5), 'float32')

        if self.transform is not None:
            input_img = self.transform(Image.fromarray(input_img[:, :, ::-1]))
        target =self.target_builder.forward(labels)
        return img_path, input_img, target

    def __len__(self):
        return self.N

class ObjDataset2(Dataset):
    # labels: [label, quality, x, y, w, h]
    def __init__(self, data, img_size=416, transform=None, transform_path=None):
        with open(data, 'rb') as f:
            self.files = pickle.load(f)
        self.imgs = []
        for key in self.files.keys():
            self.imgs.append(key)
        self.N = len(self.files)
        self.img_size = img_size
        self.transform = transform
        self.transform_path = transform_path

    def __getitem__(self, index):
        img_key = self.imgs[index % self.N]
        if self.transform_path is None:
            img_path = img_key
        else:
            img_path = self.transform_path(img_key)
        meta = self.files[img_key]
        #try:
        img = cv2.imread(img_path)
        h, w, c = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ( (pad1, pad2), (0, 0), (0, 0) ) if h <= w else ( (0, 0), (pad1, pad2), (0, 0)  )
        input_img = np.pad(img, pad, 'constant', constant_values=128)
        padded_h, padded_w, _ = input_img.shape
        input_img = cv2.resize(input_img, (self.img_size, self.img_size))
        if input_img.shape[-1] != 3:
            input_img = input_img[:, :, :3]
        if len(meta) > 0:
            labels = np.array(meta)
            cls = labels[:, 0]
            x1 = w * (labels[:, 2] - labels[:, 4] / 2)
            y1 = h * (labels[:, 3] - labels[:, 5] / 2)
            x2 = w * (labels[:, 2] + labels[:, 4] / 2)
            y2 = h * (labels[:, 3] + labels[:, 5] / 2)
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            labels = np.zeros((len(meta), 5), 'float32')
            labels[:, 4] = cls
            labels[:, 0] = x1 / padded_w
            labels[:, 1] = y1 / padded_h
            labels[:, 2] = x2 / padded_w
            labels[:, 3] = y2 / padded_h
        else:
            labels = np.zeros((2, 5), 'float32')
        '''
        except:
            print(img_path)
            input_img = 127 * np.ones((self.img_size, self.img_size, 3), 'uint8')
            labels = np.zeros((2, 5), 'float32')
        '''

        if self.transform is not None:
            input_img = self.transform(Image.fromarray(input_img[:, :, ::-1]))
        input_img = torch.from_numpy(input_img.transpose(2, 0, 1))
        #target =self.target_builder.forward(labels)
        target = labels
        return input_img, target

    def __len__(self):
        return self.N

class ListDataset(Dataset):
    def __init__(self, path, transform_path, img_size=416):
        with open(path, 'rb') as f:
            self.files = pickle.load(f)
        self.transform_path = transform_path
        self.img_shape = (img_size, img_size)
        self.img_size = img_size
        self.max_objects = 50
        self.img_files = []
        for key in self.files:
            self.img_files.append(key)

    def _construct_data(self):
        img = 127 * np.ones((self.img_size, self.img_size, 3), 'uint8')
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        labels = np.zeros((self.max_objects, 5), 'float64')
        return img, torch.from_numpy(labels)
    
    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        _img_path = self.transform_path(img_path)
        try:
            img = cv2.imread(_img_path, 1)
        except:
            img, label = self._construct_data()
            return _img_path, img, label
        labels = self.files[img_path]
        if img is None or len(labels) == 0:
            img, label = self._construct_data()
            return _img_path, img, label
        
        if img.shape[-1] != 3:
            img = img[:, :, :3]

        h, w, c = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        input_img = cv2.resize(input_img, self.img_shape)
        input_img = np.transpose(input_img, (2, 0, 1)) #Channels-first
        input_img = torch.from_numpy(input_img).float()

        labels = self.files[img_path]
        labels = np.array(labels)
        x1 = w * (labels[:, 1] - labels[:, 3] / 2)
        y1 = h * (labels[:, 2] - labels[:, 4] / 2)
        x2 = w * (labels[:, 1] + labels[:, 3] / 2)
        y2 = h * (labels[:, 2] + labels[:, 4] / 2)
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        labels[:, 1] = ((x1 + x2) / 2) / padded_w
        labels[:, 2] = ((y1 + y2) / 2) / padded_h
        labels[:, 3] *= w / padded_w
        labels[:, 4] *= h / padded_h
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[: self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return _img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)


def detection_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
