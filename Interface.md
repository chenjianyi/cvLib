# autoCV
realai CV Algorithms library
# 接口文档
# modules
## basic
*basic基于pytorch再封装的一些基本单元*
- basic.basic.convlolution2d，该接口将二维卷积、bn和激活封装在了一起
>Args:
>>in_planes:   输入的特征层数  
>>out_planes:  输出的特征层数  
>>kernel_size: 卷积核的大小  
>>stride:      卷积操作的stride，default：1  
>>padding:     卷积操作的padding，default：0  
>>dilation:    nn.Conv2d中的dilation参数，default：1  
>>groups:      nn.Conv2d中的groups参数，default：1  
>>bias: 卷积操作是否应用偏置项，default：True  
>>bn: 是否在卷积后采用batch_normalization，default：False  
>>act_fun:     激活函数的选择  
>>>"relu": nn.ReLU(inplace), inplace参数可以在接口中传入，default：False  
>>>"sigmoid": nn.Sigmoid()  
>>>"softmax": nn.Softmax(dim), dim参数可以在接口中传入，default：None
>>>"none": nn.Sequential(), 不做任何激活  
>>>*其它还可支持选项【"relu6", "rrelu", "selu", "celu", "softplus"】等，详细请参考pytorch相关激活函数文档*  
- basic.basic.residual，残差连接
>Args:
>>inp_panes: 输入的特征层数  
>>out_planes: 输出的特征层数  
>>kernel_size: 卷积核的大小  
>>stride: 第一个卷积的stride, default: 1, 第二个卷积的stride默认为1（即第二个卷积不会做下采样操作）  
>>bn: 是否在卷积后采用batch_normalization，default：True  
>>bias: 卷积操作是否应用偏置项，default：False  
- modules.module.HighResolutionModule.HighResolutionModule, 在modules.net.hrnet中构建hrnet使用，在hrnet中的每个stage构建多个branch
>Args:
>>num_branches:输出的branch数  
>>blocks:每个branch采用的block模块，可以采用BasicBlock或者Bottleneck  
>>num_blocks:每个branch采用多少个block模块进行特征提取  
>>num_inchannels:  
>>num_channels:  
>>fuse_method:  
>>multi_scale_output:  
- modules.net.stack_hourglass.StackHourglass
>Args:
>>n: 每一个Hourglass采用的scale层数  
>>nstack: 级连的数量  
>>dims: Hourglass内部每一个scale输出的特征层数, list 
>>modules: hourglass内部采用的基本模块在每个scale的重复数量，list
>>pre: 图片预处理模块，nn.Module的子类即可，可自定义，default：x4 downsample  
>>cnv_dim: 每一个hourglass最后一个scale输出的特征层数，最后一个scale一般作为网络的输出  
>>layer: hourglass内部采用的基本模块，默认为residual

>StackHourglass.forward(x)
>>input: torch.Tensor(bs, c, h, w), 例如torch.Tensor(1, 3, 512, 512)  
>>outputs: (middle_attrs, outs), middle_attrs是一个nstack x n的二维list，outs是一个长度为nstack的list

>example
```
x = torch.Tensor(1, 3, 512, 512).uniform_()  
n = 5
dims = [256, 256, 384, 384, 384, 512]
modules = [2, 2, 2, 2, 2, 4]
nstacks = 2
net = StackHourglass(n, nstacks, dims, modules)
middle_attrs, outs = net(x)
```
- modules.module.HourglassModule.HourglassModule
>Args:
>>n: Hourglass内部的scale层数  
>>dims: Hourglass内部每一个scale输出的特征层数, list  
>>modules: hourglass内部采用的基本模块在每个scale的重复数量，list  
>>layer: hourglass内部采用的基本模块, 默认为residual
- modules.block.MobileNetBlock.SeBlock，支持SwithedNorm和BatchNorm两种Norm方式
>Args:
>>inp_planes: 输入特征的channel数目  
>>reduction: 压缩因子，default：4  
>>bias: 内部卷积是否用偏置项, default=False  
>>sn, bn: defualt: sn=False, bn=True
>>>sn=False, bn=False: nn.Sequential(inp_planes)  
>>>sn=False, bn=True: nn.BatchNorm2d(inp_planes)  
>>>sn=True: SwitchNorm2d(inp_planes, using_bn=bn)  
- modules.block.MobileNetBlock.MobileNetV3Block, MobileNetV3中的block模块
>Args:  
>>kernel_size: 内部降采样卷积核的size  
>>inp_planes: 输入特征的channel数目  
>>expand_planes: InvertedResidual中扩张的channel数目  
>>out_planes: 输出特征的channel数目  
>>nolinear: 非线性激活函数  
>>semodule: SeBlock  
>>stride: 降采样卷积核的stride，stride=1时不进行降采样  
>>bias: 内部卷积是否用偏置项, default=False  
>>sn, bn: defualt: sn=False, bn=True
>>>sn=False, bn=False: nn.Sequential(inp_planes)  
>>>sn=False, bn=True: nn.BatchNorm2d(inp_planes)  
>>>sn=True: SwitchNorm2d(inp_planes, using_bn=bn) 
- modules.block.MobileNetBlock.MobileNetV2Block, MobileNetV2中的block模块
>Args:
>>in_planes: 输入特征的channel数目  
>>expansion: InvertedResidual的扩张比例  
>>out_planes: 输出特征的channel数目  
>>repeat_times: 基本模块的重复次数, 如果stride != 1, 只会有第一个基本模块进行降采样 
>>stride: 降采样卷积核的stride，stride=1时不进行降采样  
>>bias: 内部卷积是否用偏置项, default=False  
>>sn, bn: defualt: sn=False, bn=True
>>>sn=False, bn=False: nn.Sequential(inp_planes)  
>>>sn=False, bn=True: nn.BatchNorm2d(inp_planes)  
>>>sn=True: SwitchNorm2d(inp_planes, using_bn=bn)  
- modules.net.mobilenet.MobileNetV3_Large, 支持SwithedNorm和BatchNorm两种Norm方式
>Args:  
>>output: "classifier"或者"feature"  
>>>"classfier"：分类器，输出分类结果，可以选择传入num_classes参数，num_classes默认为1000  
>>>"feature"：特征提取器，输出MobileNetV3Block每个不同scale的特征，list:【4Xdownsample,8Xdownsample,16Xdownsample,32Xdownsample】   

>>bias: 内部卷积是否用偏置项, default=False  
>>sn, bn: defualt: sn=False, bn=True
>>>sn=False, bn=False: nn.Sequential(inp_planes)  
>>>sn=False, bn=True: nn.BatchNorm2d(inp_planes)  
>>>sn=True: SwitchNorm2d(inp_planes, using_bn=bn)  

>example:
```
net = MobileNetV3_Large(output='feature', bias=True, sn=True, bn=True)
x = torch.Tensor(2, 3, 300, 300).uniform_()
outs = net(x)
###outs: [torch.Size([2, 24, 75, 75]),torch.Size([2, 40, 38, 38]),torch.Size([2, 160, 19, 19]),torch.Size([2, 160, 10, 10])]
```
- modules.net.mobilenet.MobileNetV3_Small, 支持SwithedNorm和BatchNorm两种Norm方式
>Args:  
>>output: "classifier"或者"feature"  
>>>"classfier"：分类器，输出分类结果，可以选择传入num_classes参数，num_classes默认为1000  
>>>"feature"：特征提取器，输出MobileNetV3Block每个不同scale的特征，list:【4Xdownsample,8Xdownsample,16Xdownsample,32Xdownsample】   

>>bias: 内部卷积是否用偏置项, default=False  
>>sn, bn: defualt: sn=False, bn=True
>>>sn=False, bn=False: nn.Sequential(inp_planes)  
>>>sn=False, bn=True: nn.BatchNorm2d(inp_planes)  
>>>sn=True: SwitchNorm2d(inp_planes, using_bn=bn)  
- modules.net.mobilenet.MobileNetV2, 支持SwithedNorm和BatchNorm两种Norm方式
>Args:  
>>output: "classifier"或者"feature"  
>>>"classfier"：分类器，输出分类结果，可以选择传入num_classes参数，num_classes默认为1000  
>>>"feature"：特征提取器，输出MobileNetV2Block每个不同scale的特征，list:【4Xdownsample,8Xdownsample,16Xdownsample,32Xdownsample】  

>>bias: 内部卷积是否用偏置项, default=False  
>>sn, bn: defualt: sn=False, bn=True
>>>sn=False, bn=False: nn.Sequential(inp_planes)  
>>>sn=False, bn=True: nn.BatchNorm2d(inp_planes)  
>>>sn=True: SwitchNorm2d(inp_planes, using_bn=bn)  
