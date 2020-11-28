'''
This code is to use Detectron2 API for model evaluation on customized dataset.
'''
# basic library
import torch
import torchvision
import numpy as np
import os
import json
import cv2
import random
import pdb
import matplotlib.pyplot as plt
# detectron2 library
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

setup_logger()

# main function:
if __name__ == '__main__':
    # image/annotation file paths
    val_annotation_file = "../datasets/annotations/instances_val2017.json"
    val_image_path = "../datasets/val2017"
    val_dataset_name = "my_dataset_val"

    # register dataset to coco format
    print("Register dataset......")
    register_coco_instances(val_dataset_name, {}, val_annotation_file, val_image_path)

    # setup training configuration
    print("Setup configuration......")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.DEVICE = 'cpu' # select either cpu or gpu devices

    # setup prediction and evalaution
    print("Setup predictor and evaluator......")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, val_dataset_name)
    trainer = DefaultTrainer(cfg)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))