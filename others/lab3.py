from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, dataloader
from PIL import Image
import json

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

def _default_loader(path):#PIL图片加载器
    img = Image.open(path).convert('RGB')
    print(type(img))
    transform1 = transforms.Compose([transforms.ToTensor(),])
    return transform1(img)

coco_demo = COCODemo(
    cfg,
    min_image_size=18000,
    confidence_threshold=0.7,
)
# load image and then run prediction
if __name__ == "__main__":
    image = _default_loader('./haha.jpg')
    print("image:",image.size())
    predictions = coco_demo.run_on_opencv_image(image)
    print(type(predictions))
    print("hello")