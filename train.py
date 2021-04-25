"""
This code is to use Detectron2 API to do model training on customized dataset.
"""
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
from detectron2.engine import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm

setup_logger()

# customized coco trainer to include evalution on testset
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                "val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(
                    total_val_loss=losses_reduced, **loss_dict_reduced
                )


# main function:
if __name__ == "__main__":
    # image/annotation file paths
    train_annotation_file = "./datasets/annotations/instances_train2017.json"
    val_annotation_file = "./datasets/annotations/instances_val2017.json"
    train_dataset_name = "COCO2017_train"
    val_dataset_name = "COCO2017_val"
    train_image_path = "./datasets/train2017"
    val_image_path = "./datasets/val2017"

    # register dataset to coco format
    print("Register dataset......")
    register_coco_instances(
        train_dataset_name, {}, train_annotation_file, train_image_path
    )
    register_coco_instances(val_dataset_name, {}, val_annotation_file, val_image_path)

    # get metadata
    print("Retrieve metadata......")
    train_metadata = MetadataCatalog.get(train_dataset_name)
    train_dic = DatasetCatalog.get(train_dataset_name)

    # one sample visualization
    print("Sample visualizer of dataset......")
    img = cv2.imread(train_dic[0]["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=1.2)
    vis = visualizer.draw_dataset_dict(train_dic[0])
    #plt.imshow(vis.get_image())
    #plt.show()

    # setup training configuration
    print("Setup configuration......")
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.VAL = (val_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 50
    cfg.SOLVER.STEPS = [40, ]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.DEVICE = "cpu"  # select either cpu or gpu devices
    cfg.TEST.EVAL_PERIOD = 2

    # start training
    print("Start training......")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # trainer = CocoTrainer(cfg) # or use CocoTrainer to include evaluation
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()
