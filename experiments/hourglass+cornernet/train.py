import os
import json
import numpy as np
import random
import argparse
import os

import torch
from torch.utils.data import DataLoader

from my_dataset import Dataset
from models import Model


#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="CornerNet")
parser.add_argument('--image_meta', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/data_seg2_2_cornernet.pkl', help='train file (pickle format)')
#parser.add_argument('--image_meta', type=str, default='./meta/single_800.pkl')
parser.add_argument('--label_file', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/label_file.txt', help='label file (txt format)')
parser.add_argument('--pretrained_model', type=str, default='./weights/CornerNet_500000.pkl')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--gpu', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=600)
parser.add_argument('--out_dir', type=str, default='cornernet')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)


def train():
    out_dir = os.path.join('/data/chenjianyi/records/cvLib', opt.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dataset = Dataset(opt.image_meta, opt.label_file)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    with open(opt.label_file, 'r') as f:
        label_list = f.readlines()
    label_list = [item.strip() for item in label_list]
    label2id_dict = {k: v for v, k in enumerate(label_list)}
    id2label_dict = {k: v for k, v in enumerate(label_list)}
    num_classes = len(label_list)

    n = 5
    dims = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    nstacks = 2
    nnet = Model(nstacks, n, dims, modules, num_classes)
    nnet.train()
    nnet.cuda()

    """
    if opt.pretrained_model is not None and opt.pretrained_model != '':
        if not os.path.exists(opt.pretrained_model):
            raise ValueError('pretrained model not exists')
        print('loading from pretrained model')
        nnet.load_pretrained_params(opt.pretrained_model, ignore=True)
    """

    learning_rate = opt.learning_rate
    print('setting learning rate to: {}'.format(learning_rate))
    if opt.optimizer:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, nnet.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )

    print('training start...')
    for epoch in range(opt.max_epoch):
        loss_tmp, focal_loss_tmp, pull_loss_tmp, push_loss_tmp, regr_loss_tmp = 0., 0., 0., 0., 0.
        for step, data in enumerate(dataloader):
            x = data['xs'].cuda()
            targets = [item.cuda() for item in data['ys']]
            targets = [targets] * nstacks

            loss, focal_loss, pull_loss, push_loss, regr_loss = nnet(x, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 0:
                loss_tmp, focal_loss_tmp, pull_loss_tmp, push_loss_tmp, regr_loss_tmp = \
                    loss, focal_loss, pull_loss, push_loss, regr_loss
            else:
                loss_tmp = 0.5 * loss_tmp + 0.5 * loss.item()
                focal_loss_tmp = 0.5 * focal_loss_tmp + 0.5 * focal_loss.item()
                pull_loss_tmp = 0.5 * pull_loss_tmp + 0.5 * pull_loss.item()
                push_loss_tmp = 0.5 * push_loss_tmp + 0.5 * push_loss.item()
                regr_loss_tmp = 0.5 * regr_loss_tmp + 0.5 * regr_loss.item()
            if step % 1 == 0:
                print('[%d, %d/%d] [loss: %f, focal_loss: %f, pull_loss: %f, push_loss: %f, regr_loss: %f]' % \
                      (epoch+1, step, len(dataloader), loss_tmp, focal_loss_tmp, pull_loss_tmp, push_loss_tmp, regr_loss_tmp))

        if epoch % 5 == 0 and epoch > 0:
            path = os.path.join(out_dir, 'weight_%s.pth' % str(epoch+1))
            nnet.save_params(path)

        if epoch % 50 == 0 and epoch > 0:
            print('setting learning rate to: {}'.format(learning_rate))
            learning_rate /= 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

if __name__ == '__main__':
    train()
