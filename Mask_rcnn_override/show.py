import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.config import Config

 
# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
# config.display()

# Load model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Labels 
class_names = ['BG','person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush', 'swimming pool',
               'towel','FitnessBike', 'Treadmill']

#影像路徑
img_path = './test5.jpg'

#偵測類別
useful_class_names = ['person']

#過濾類別涵式
def filter_r(r):
    # filter usefule result
    filter_idex = []
    for i,clss_id in enumerate(r['class_ids']):
        if class_names[clss_id] in useful_class_names:
            filter_idex.append(i)

    r['rois'] = np.array([_ for i,_ in enumerate(r['rois']) if i in filter_idex])
    r['masks'] = np.concatenate([r['masks'][...,i:i+1] for i,_ in enumerate(r['masks']) if i in filter_idex],axis = -1)
    r['class_ids'] = np.array([_ for i,_ in enumerate(r['class_ids']) if i in filter_idex])
    r['scores'] = np.array([_ for i,_ in enumerate(r['scores']) if i in filter_idex])
	
    return r

# Load image
image = skimage.io.imread(img_path)
# Run detection
results = model.detect([image], verbose=1)
# Visualize results
r = results[0]
# Filter Label
rr = filter_r(r)
# Show image and save image
visualize.display_instances(image, rr['rois'], rr['masks'], rr['class_ids'], 
                            class_names, rr['scores'], figsize=(8, 8), show_bbox = True)
y1, x1, y2, x2 = visualize.display_instances_save(image, rr['rois'], rr['masks'], rr['class_ids'], 
                            class_names, rr['scores'], figsize=(8, 8), show_bbox = False)
print("y1 :", y1)
print("x1 :", x1)
print("y2 :", y2)
print("x2 :", x2)