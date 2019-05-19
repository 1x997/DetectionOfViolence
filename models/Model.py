"""
总模型
[image] ----> Faster R-CNN ----> SHDL ----> SVM ----> [output]
注意：该模型用于最终的实际应用中
"""
from models import SVM, FRCNN, SHDL
from data import Dataset
import torch
import torch.nn as nn
import time

class MODEL():
    def __init__(self):
        self.TARGET_DETECTION = FRCNN.RCNN(load = True)     #目标检测
        self.JOINT_EXTRACT = SHDL.SHDL(load = True)         #关节提取
        self.CLASSIFIER = SVM.SVM(load = True)              #分类

    def __call__(self, image):
        """
        输入任意尺寸的图片
        返回判断数据
        """
        #step1 : 输入任意尺寸的图片，输出为分割后的区域Tensor列表
        start = time.time()
        output1 = self.TARGET_DETECTION(image)
        print("图像分割已完成，用时:{0}s".format(start - time.time()))

        #output2 : 输入为Tensor格式的图像区域列表，输出为存储关节信息的列表
        start = time.time()
        output2 = [self.JOINT_EXTRACT(i) for i in output1]
        print("关节提取已完成，用时:{0}s".format(start - time.time()))

        #output3 : 输入为存储关节信息的列表，输出为判断动作类型的列表
        start = time.time()
        output3 = [self.CLASSIFIER(i) for i in output2]
        print("动作判断已完成，用时:{0}s".format(start - time.time()))

        return output3

if __name__ == "__main__":
    ourMODEL = MODEL()
    print(ourMODEL)
