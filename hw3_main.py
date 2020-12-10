# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 22:36:19 2020

@author: kuanyu
"""

# In[]
# import, setup, and register dataset
import os
import cv2
import json
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from pycocotools.coco import COCO
from itertools import groupby
from pycocotools import mask as maskutil


# provided function
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(
            groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0],
                                          rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


setup_logger()

# read and register the trainset
train_path = 'hw3dataset\\train_images'
json_path = os.path.join(train_path, 'pascal_train.json')
register_coco_instances("hw3_trainset", {}, json_path, train_path)
train_metadata = MetadataCatalog.get("hw3_trainset")

# In[]
# ========== train ==========
cfg = get_cfg()
cfg.merge_from_file(
    "configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("hw3_trainset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "x-101-32x8d.pkl"  # pre-trained model
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.SOLVER.CHECKPOINT_PERIOD = 10000

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# In[]
# ========== test ==========
cfg = get_cfg()
cfg.merge_from_file(
    "configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'x-101-32x8d', "model_0079999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.DATASETS.TEST = ("hw3_trainset",)
predictor = DefaultPredictor(cfg)

cocoGt = COCO("hw3dataset/test.json")
res_path = 'test_result_005'
os.makedirs(res_path, exist_ok=True)
coco_dt = []
for imgid in cocoGt.imgs:
    filename = cocoGt.loadImgs(ids=imgid)[0]['file_name']
    im = cv2.imread("hw3dataset/test_images/" + filename)
    print('predicting ' + filename)
    outputs = predictor(im)
    # v = Visualizer(im[:, :, ::-1],
    #                metadata=train_metadata,
    #                scale=2,
    #                )
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # res_file = os.path.join(res_path, filename)
    # cv2.imwrite(res_file, v.get_image()[:, :, ::-1])

    anno = outputs['instances'].to('cpu').get_fields()
    masks = anno['pred_masks'].numpy()
    classes = anno['pred_classes'].numpy()
    scores = anno['scores'].numpy()

    n_instances = len(scores)
    if len(classes) > 0:  # If any objects are detected in this image
        for i in range(n_instances):  # Loop all instances
            pred = {}
            pred['image_id'] = imgid
            pred['category_id'] = int(classes[i]) + 1
            pred['segmentation'] = binary_mask_to_rle(masks[i, :, :])
            pred['score'] = float(scores[i])
            coco_dt.append(pred)

with open("309551107_007.json", "w") as f:
    json.dump(coco_dt, f)

