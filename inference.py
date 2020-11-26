'''
This code is for using Detectron2 API to run inference on the customized dataset.
'''
# basic library
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os
import json
import cv2
import random
import matplotlib.pyplot as plt
import pdb
# detectron2 library
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()

# main function:
if __name__ == '__main__':
    # read images
    img = cv2.imread("../datasets/val2017/000000000139.jpg")

    # setup configurations
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = 'cpu'

    # setup predictor
    predictor = DefaultPredictor(cfg)

    # run inference
    outputs = predictor(img)

    # show output
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # visualization
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image())
    plt.show()