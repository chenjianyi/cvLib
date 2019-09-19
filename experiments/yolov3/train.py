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
#from models import Model
from darknet53_fpn_yolov3 import Darknet53_FPN_YoloV3
from mobilenetv3_fpn_yolov3 import MobileNetV3_FPN_YoloV3
from m2det_fpn_yolov3 import M2det_FPN_YoloV3
from hrnet_fpn_yolov3 import Hrnet_FPN_YoloV3
import numpy as np

from _utils import weights_init_normal
from _utils import TargetBuilder, TargetBuilder2
from dataset import ObjDataset, ObjDataset2

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=600, help='number of epoches')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
parser.add_argument('--image_meta', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/VOC2012_train.pkl', help='the path of train set file')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--weights_path', type=str, default='', help='optional. path to weights file')
parser.add_argument('--label_file', type=str, default='/data/chenjianyi/project/guangfu/yolov3_pytorch/meta/VOC_label_file.txt', help='mapping of label_index to label_name')
parser.add_argument('--backbone', type=str, default='hrnet')

parser.add_argument('--confidence_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou threshold for non-maxinum suppression')
parser.add_argument('--checkpoint_interval', type=int, default=5, help='how often to save model weights')
parser.add_argument('--checkpoint_dir', type=str, default='cvLib/hrnet+fpn+yolov3', help='dir to save model')
parser.add_argument('--gpu', type=str, default='3', help='which device to use for train')
opt = parser.parse_args()
print(opt)

checkpoint_dir = os.path.join('/data/chenjianyi/records/', opt.checkpoint_dir)
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
anchors1 = [[10,13], [16,30], [33,23]]
anchors2 = [[30,61], [62,45], [59,119]]
anchors3 = [[116,90], [156,198], [373,326]]
anchors = [anchors1, anchors2, anchors3]
img_dim = 416
g_dims = [52, 26, 13]

hp = {}
hp['momentum'] = 0.9
hp['weight_decay'] = 0.0001
hp['lr'] = float(opt.learning_rate)

#Get dataloader
target_builder = TargetBuilder2(anchors, num_classes, dims=g_dims, img_size=img_dim)
def path_transform(s):
    return s
transform_train = transforms.Compose([
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.ToTensor()
])
dataset = ObjDataset2(opt.image_meta, img_size=img_dim, target_builder=target_builder,
    transform=transform_train, transform_path=path_transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=12)
num_batches = len(loader)

#Initiate model
print('Initiate model...')
if opt.backbone == 'darknet53':
    Model = Darknet53_FPN_YoloV3
elif opt.backbone == 'mobilenetv3':
    Model = MobileNetV3_FPN_YoloV3
elif opt.backbone == 'm2det':
    Model = M2det_FPN_YoloV3
elif opt.backbone == 'hrnet':
    Model = Hrnet_FPN_YoloV3
model = Model(num_classes, anchors, num_levels, img_dim, g_dims)
model.apply(weights_init_normal)
if opt.weights_path and os.path.exists(opt.weights_path):
    state_dict = torch.load(opt.weights.path)
    model.load_state_dict(state_dict)

print('Model to cuda...')
model = model.cuda()
model.train()

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
    for (_, imgs, targets) in loader:
        imgs = Variable(imgs.type(Tensor))
        targets = [Variable(target.type(Tensor), requires_grad=False) for target in targets]
        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        loss_xy = 0.9 * loss_xy + 0.1 * model.losses['xy']
        loss_wh = 0.9 * loss_wh + 0.1 * model.losses['wh']
        loss_conf = 0.9 * loss_conf + 0.1 * model.losses['conf']
        loss_cls = 0.9 * loss_cls + 0.1 * model.losses['cls']
        n_shot = 0.9 * n_shot + 0.1 * model.losses['recall']

        batch_i += 1
        if batch_i % 20 == 0:
            print('[%d, %d/%d, %.1f] [xy %.5f, wh %.5f, conf %.5f, cls %.5f, box %.1f]' %(
                epoch+1, batch_i, num_batches, time.time()-_t, loss_xy, loss_wh, loss_conf, loss_cls, n_shot))
            _t = time.time()
    
    if (epoch + 1) % opt.checkpoint_interval == 0 and epoch > 50:
        current = time.strftime("%Y%m%d%H", time.localtime(time.time()))
        path = '{}/{}_{}.weights'.format(checkpoint_dir, current, epoch+1)
        torch.save(model.state_dict(), path)
    
