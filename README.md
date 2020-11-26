# detectron2-pipelines
Customized pipelines using Detectron2 API developed by Facebook AI Research [1].

## Installation
If you are using MacOS:
```
git clone https://github.com/facebookresearch/detectron2.git

cd detectron2

MACOSX_DEPLOYMENT_TARGET=10.X.X CC=clang CXX=clang++ pip install -e .
```
Else follow the steps in https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

## Useful preprocessing tools
1. Converting PASCAL VOC xml labels to COCO json format [4].

## References
[1] https://github.com/facebookresearch/detectron2

[2] https://blog.csdn.net/m0_37709262/article/details/102732057

[3] https://towardsdatascience.com/how-to-train-detectron2-on-custom-object-detection-data-be9d1c233e4

[4] https://github.com/roboflow-ai/voc2coco
