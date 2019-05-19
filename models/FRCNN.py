"""
Faster R-CNN模型：
    介绍：输入图片自动转化成224*224大小，输出类别
"""

#from . import FPN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
import cv2
import torchvision.models as models

class RCNN(nn.Module):
    def __init__(self, load = False):
        if not load:
            super(RCNN, self).__init__()
            self.vgg16 = self._make_vgg16() #VGG层，自动将输入的图片转化为224*224
            self.convLayer = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
            self.conv1x1 = nn.Conv2d(512, 18, kernel_size=1, stride=1)
        else:
             return

    def _load(self):
        pass

    def _make_vgg16(self):
        model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
        model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        # model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行
        return model

    def ImageInit(self, image):
        image = image.view(1, 3, image.size[0], image.size[1])
        return image

    def forward(self, input):
        """
        [input] ----> vgg16 ----> [feature map] ----> convLayer, relu ----> conv1*1
                                        |                      |
                                        |                      |
                                        |                      v
                                        v
        """
        x = self.ImageInit(input) #初始化
        x = self.vgg16(x) #feature map
        y = F.relu(self.convLayer(x))
        y1 = self.conv1x1(y)


if __name__ == "__main__":
    import torchvision.models as models
    model = models.vgg16(pretrained=True)
