# YoloV3-Lite
This YoloV3-Lite repository contains code for YoloV3-set algorithm. Generally, YoloV3-Lite shares a common head, but not the backbones. In our experiments, YoloV3 is not working(difficylt for training) without FPN. So our YoloV3-Lite contains backbone, fpn and head.


## Usage
- Dependencies:
    - python >= 3.5
    - torch >= 1.0
    - cv2  

- Algorithms:
    - Darknet53 + FPN + YoloV3
    - MobilenetV3 + FPN + YoloV3
    - M2det + FPN + YoloV3
    - Hrnet + FPN + YoloV3
    - VGG + FPN + YoloV3 (to do)
    - Resnet + FPN + YoloV3 (to do)
    - Densenet + FPN + YoloV3 (to do)

- To train a model, run `python3 train.py`.

    Arguments:
    - `--epoch <epoch(int)>` the number of epoches to train
    - `--learning_rate <learning_rate(float)>` the learning rate for training
    - `--image_meta <pickle_file_path(str)>` the training meta file, the data format is showing in Data below.
    - `--batch_size <batch_size(int)>` the batch size for training
    - `--weights_path <weights_path(str)>` the initial weights file, if no, just set '' string
    - `--label_file <label_file_path(str)>` this file is for the mapping of label to index, a txt file with one label name in one line
    - `--backbone <backbone(str)>` supporting `darknet53`, `mobilenetv3`, `m2det`, 'hrnet'
    - `--confidence_thers` not used in training stage
    - `--nms_thres` not used in training stage
    - `--checkpoint_dir <checkpoint_dir(str)>` the suffix dir to save a checkpoint file
    - `--gpu <gpu_id(int)>` which device to train  

    Data:
    - `image_meta`: pickle file. A dict. Key is the image path, and value is a 2d list (list of BBox). BBox is a 6-items list, [cls, quality, x, y, w, h, cls], (x, y) is the centre point of BBox, (w, h) is the width and height of BBox, cls is class label of BBox, quality(set to 1) is reversed field. x, y, w, h is normalized to the width and height of image. 
       {path1: [[cls, quality, x, y, w, h], [...], [...], ...], path2: ...}

Here is a example to call:

```
    python3 train.py --gpu 0 --image_meta VOC2012_train.pkl --label_file VOC_label_file.txt --backbone mobilenetv3
```

