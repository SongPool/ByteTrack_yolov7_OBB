## ByteTrack-YOLOV7-OBB：带目标框旋转的ByteTrack


在bytetrack框架中使用旋转目标检测，只需要训练yolo模型即可使用

修改思路为使用旋转检测结果的最小正外接矩形作为bbox输入到bytetrack中

## 以下为目标检测模型训练文档
### 相关仓库
| 目标检测模型 | 路径 |
| :----- | :----- |
YoloV7-OBB | https://github.com/Egrt/yolov7-obb
YoloV7-Tiny-OBB | https://github.com/Egrt/yolov7-tiny-obb

### 性能情况
| 训练数据集 | 权值文件名称	| 测试数据集 | 输入图片大小 | mAP 0.5 |
| :-----: | :------: | :------: | :------: | :------: |
| SSDD | [yolov7_obb_ssdd.pth](https://github.com/Egrt/yolov7-obb/releases/download/V1.0.0/yolov7_obb_ssdd.pth) | SSDD-Val | 640x640 | 95.22

### 所需环境
torch==1.10.1
torchvision==0.11.2
lap==0.4.0
lap容易有和numpy有版本问题，建议先装
```python
# roate_nms 拓展安装
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

### 文件下载
SSDD数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/1Lpg28ZvMSgNXq00abHMZ5Q
提取码: 2021

### 训练步骤
#### a、训练VOC07+12数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07+12的数据集，解压后放在根目录**  

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   
生成的数据集格式为image_path, x1, y1, x2, y2, x3, y3, x4, y4(polygon), class。 

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

#### b、训练自己的数据集
1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```python
cat
dog
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，大家可以在下载库后仔细看注释，其中最重要的部分依然是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！训练自己的数据集必须要修改！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

### 预测步骤
#### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权值，放入model_data，运行predict.py
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
#### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolov7_weights.pth',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [640, 640],
    #------------------------------------------------------#
    #   所使用到的yolov7的版本，本仓库一共提供两个：
    #   l : 对应yolov7
    #   x : 对应yolov7_x
    #------------------------------------------------------#
    "phi"               : 'l',
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : True,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## bytetrack实现目标跟踪
修改好yolo.py的模型权重后，运行demo_track.py（添加自己的检测视频路径即可）

## Reference
### https://github.com/tungdop2/ByteTrack_yolov5
### https://github.com/Egrt/yolov7-obb
