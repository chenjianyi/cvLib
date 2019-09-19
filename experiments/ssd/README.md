# SSD-Lite
This SSD-Lite repository contains code for SSD-set algorithm. Generally, SSD-Lite shares a common head, but not the backbones. Our SSD-Lite contains backbone and head.


## Usage
- Dependencies:
    - python >= 3.5
    - torch >= 1.0
    - cv2  

- Algorithms:
    - DSOD(densenet + SSD)
    - SSD300(VGGSSD300 + SSD)
    - SSD512(VGGSSD512 + SSD): remained experiments
    - M2det(M2det + SSD)
    - FPN152(ResnetFPN152 + SSD): remained experiments
    - FPN152(ResnetFPN101 + SSD): to to and remained experiments
    - FPN152(ResnetFPN50 + SSD): to do and remained experiments
    - Hrnet + SSD: to do

- To train a model, run `python3 train.py`.

    Arguments:
    - `--epoch <epoch(int)>` the number of epoches to train
    - `--learning_rate <learning_rate(float)>` the learning rate for training
    - `--image_meta <pickle_file_path(str)>` the training meta file, the data format is showing in Data below.
    - `--batch_size <batch_size(int)>` the batch size for training
    - `--weights_path <weights_path(str)>` the initial weights file, if no, just set '' string
    - `--label_file <label_file_path(str)>` this file is for the mapping of label to index, a txt file with one label name in one line
    - `--model <backbone(str)>` supporting `m2det`, `dsod`, `vgg_ssd300`, `vgg_ssd512`, `resnet_fpn152`
    - `--confidence_thers` not used in training stage
    - `--nms_thres` not used in training stage
    - `--checkpoint_dir <checkpoint_dir(str)>` the suffix dir to save a checkpoint file
    - `--gpu <gpu_id(int)>` which device to train  

    Data:
    - `image_meta`: pickle file. A dict. Key is the image path, and value is a 2d list (list of BBox). BBox is a 5-items list, [cls, quality, x, y, w, h, cls], (x, y) is the centre point of BBox, (w, h) is the width and height of BBox, cls is class label of BBox, quality(set to 1) is reversed field. x, y, w, h is normalized to the width and height of image. 
       {path1: [[x, y, w, h, cls], [...], [...], ...], path2: ...}

Here is a example to call:

```
    python3 train.py --gpu 0 --image_meta VOC2012_train.pkl --label_file VOC_label_file.txt --model m2det
```

