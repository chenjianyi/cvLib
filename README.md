CVLib is an object detection toolbox based on Pytorch. Generally, we divide an object detection algorithm into two parts: backbone and head. Backbone is a feature extractor, maybe multi-scale features. Usually, Backbone network may conbine with a FPN block to enhance the features. Head is focusing on object detection algorithm itself. Head contains targets builder, loss, decoder and so on. It's very convenient to build a new algorithm just picking a backbone from our modules.net and picking a head from our heads. We also packed many useful blocks, all of which are in our modules.basic, modules.block, modules.module, so it will be so convenient and so quickly to use, such as building a new backbone network. We also provide some series alrorithms in our experiments, such as Cornetnet-Lite, SSD-Lite, Yolo-Lite.



## Usage
- Dependencies:
    - python >= 3.5
    - torch >= 1.0
    - cv2
- Compile ops:
```
    cd modules/ops/_cpools
    python3 setup.py install --user
```

# Algorithms
- SSD-Lite
    - refering to [experiments/ssd/README.txt]

- Yolo-Lite
    - refering to [experiments/yolo/README.txt]

- Cornetnet-Lite
    - to do

# interface document
- refering to [接口文档.md]
