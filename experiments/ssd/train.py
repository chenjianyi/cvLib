import os
import sys
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from m2det_ssd import M2det_SSD
from densenet_ssd import Densenet_SSD
from ssd300 import SSD300
from ssd512 import SSD512
from fpn152 import FPN152
import numpy as np

from utils2 import weights_init_normal
from utils2 import TargetBuilder, TargetBuilder2
from dataset import ObjDataset2, detection_collate

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='number of epoches')
parser.add_argument('--learning_rate', type=int, default=0.001, help='learning_rate')
parser.add_argument('--image_meta', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/VOC2012_train.pkl', help='the path of train set file')
parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
parser.add_argument('--weights_path', type=str, default='', help='optional. path to weights file')
parser.add_argument('--label_file', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/VOC_label_file.txt', help='mapping of label_index to label_name')
parser.add_argument('--model', type=str, default='m2det')

parser.add_argument('--confidence_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou threshold for non-maxinum suppression')
parser.add_argument('--checkpoint_interval', type=int, default=5, help='how often to save model weights')
parser.add_argument('--checkpoint_dir', type=str, default='m2det+ssd', help='dir to save model')
parser.add_argument('--gpu', type=str, default='1', help='which device to use for train')
opt = parser.parse_args()
print(opt)

checkpoint_dir = os.path.join('/data/chenjianyi/records/cvLib', opt.checkpoint_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
torch.backends.cudnn.benchmark = True
if not os.path.exists(checkpoint_dir):
    os.system('mkdir %s' % checkpoint_dir)

with open(opt.label_file, 'r') as f:
    label_list = f.readlines()
label_list = [item.strip() for item in label_list]
label2id_dict = {k: v for v, k in enumerate(label_list)}
id2label_dict = {k: v for k, v in enumerate(label_list)}

num_classes = len(label_list)
num_levels = 8
num_scales = 6
num_anchors = 6
img_size = 320

hp = {}
hp['momentum'] = 0.9
hp['weight_decay'] = 0.0001
hp['lr'] = float(opt.learning_rate)

#Initiate model
print('Initiate model...')
if opt.model == 'm2det':
    Model = M2det_SSD
elif opt.model == 'dsod':
    Model = Densenet_SSD
elif opt.model == 'vgg_ssd300':
    Model = SSD300
    img_size = 320
elif opt.model == 'vgg_ssd512':
    Model = SSD512
    num_scales = 7
    img_size = 512
elif opt.model == 'resnet_fpn152':
    Model = FPN152
    num_scales = 7
    img_size = 512
model = Model(num_classes, num_scales, num_anchors, num_levels)
if opt.weights_path and opt.weights_path != '':
    model.load_weights(opt.weights_path)
    #model.transfer_weights(opt.weights_path, 80)

print('Model to cuda...')
model = model.cuda()
model.train()

#Get dataloader
dataset = ObjDataset2(opt.image_meta, img_size=img_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, collate_fn=detection_collate)
num_batches = len(loader)

#Training
print('Start training...')
Tensor = torch.cuda.FloatTensor
optimizer = optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'], dampening=0, weight_decay=hp['weight_decay'])
loss_xy, loss_wh, loss_conf, loss_cls, loss_total, n_shot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
index = np.arange(opt.batch_size)
start_epoch = 0
for epoch in range(start_epoch, opt.epoch):
    if epoch + 1 in [int(opt.epoch * 0.15), int(opt.epoch * 0.75)]:
        for group in optimizer.param_groups:
            group['lr'] /= 10
    batch_i, _t = 0, time.time()
    for (imgs, targets) in loader:
        imgs = Variable(imgs.type(Tensor))
        targets = [Variable(target.type(Tensor), requires_grad=False) for target in targets]
        optimizer.zero_grad()
        loss, loss_l, loss_c = model(imgs, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        batch_i += 1
        if batch_i % 20 == 0:
            print('[%d, %d/%d, %.1f] [loss %.4f, loss_l %.4f, loss_c %.4f]' %(
                epoch+1, batch_i, num_batches, time.time()-_t, loss, loss_l, loss_c))
            _t = time.time()
    
    if (epoch + 1) % opt.checkpoint_interval == 0 and epoch > 50:
        current = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        path = '{}/{}_{}.weights'.format(checkpoint_dir, current, epoch+1)
        torch.save(model.state_dict(), path)
    
