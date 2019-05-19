"""
COCO数据集说明：
        每个.json注释文件加载后为一个字典，字典有五个(或四个)组成部分：info\licenses\images\annotations\categories
        其中info为该文件的总信息，为字典，包括数据集名称、url、版本、年份
            eg : {'description': 'COCO 2017 Dataset', 'url': 'http://cocodataset.org', 'version': '1.0', 'year': 2017, 'contributor': 'COCO Consortium', 'date_created': '2017/09/01'}
        licenses为该文件的若干个许可证组成的列表，每个许可证信息为一字典，包括url、id、名称
            eg : {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}
        images为每个图片的信息组成的列表，每份信息为字典，包括许可证编号、文件名称、url、尺寸、时间、id
            eg : {'license': 2, 'file_name': '000000413247.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000413247.jpg', 'height': 426, 'width': 640, 'date_captured': '2013-11-23 19:55:57', 'flickr_url': 'http://farm5.staticflickr.com/4065/4568885003_f08cd0bbfa_z.jpg', 'id': 413247}
        annotations为每个图片对应的注释信息组成的列表，每份注释信息为一字典，不同文件中注释信息也不同
        categories为每个图片的类别信息，每份信息包括supercategories(大类)、name(小类)、id
COCO_Dataset数据集类使用说明 :
        初始化 : 输入数据集加载器（默认为COCO数据集加载器）、转换器（默认为ToTensor转换器）、图片加载器（默认为PIL加载器）
            eg : myDataset = COCO_Dataset()
        索引 : [rank]，返回第rank个图片的矩阵和标签
        取长 : len()，返回图片总数
        自定义数据集加载器 : 需要满足要求：返回格式为列表，列表中每个元素为一个元素的字典信息，字典索引包含'path':<图片路径>。
                                eg : [{'path'='E:\\COCO\\001.jpg', 'class':'person'}, ...]
                          调用COCO_Dataset.reLoad方法，输入新加载器，更新数据
"""
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

"""
PIL图片加载器：输入路径，返回PIL图片格式
"""
def _default_loader(path):#PIL图片加载器
    return Image.open(path).convert('RGB')

"""
COCO数据集加载器：路径从配置文件读取，返回列表
"""
def _cocoLoader():
    with open("..\\data\\config.json", "r") as f:
        json.load(f)
    pass

class COCO_Dataset(Dataset):
    def __init__(self, datasetLoader = _cocoLoader(), transform=transforms.ToTensor(), loader=_default_loader):
        self.images = datasetLoader() #列表，记录图片路径名称和标签
        self.transform = transform #转换器
        self.loader = loader #加载器，默认是PIL加载器

    def __getitem__(self, index):
        fn= self.images[index]['path']
        label = {i:self.images[index][i] for i in self.images[index] if i != 'path'}
        img = self.loader(fn) #加载图片
        if self.transform is not None: #转换
            img = self.transform(img)
        return img,label    #返回转换后的图片和标签

    def __len__(self):
        return len(self.images) #返回图片数量

    def reLoad(self, new_datasetLoader):
        self.images = new_datasetLoader()

    @staticmethod
    def showImage(image):
        """
        :param image: 输入Tensor格式的图像或PIL格式的图像，否则不显示
        :return: 显示图像
        """
        if(type(image).__name__=='Tensor'):
            image = transforms.ToPILImage()(image)  # 转化成PIL图像
        elif(type(image).__name__=='JpegImageFile'):
            pass
        else:
            print("error")
            return
        plt.imshow(image)
        plt.show()