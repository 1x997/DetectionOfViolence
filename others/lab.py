from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def readImage(path, size=256):
    mode = Image.open(path)
    print(type(mode).__name__=='JpegImageFile')
    transform1 = transforms.Compose([
        transforms.Resize(size),
        #transforms.CenterCrop((20,20)),
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    return mode


def showTorchImage(image):
    mode = transforms.ToPILImage()(image) #转化成PIL图像
    plt.imshow(mode)
    plt.show()


if __name__ == '__main__':
    mode = readImage(path = '../data/faces/1198_0_861.jpg',size=224)
    showTorchImage(mode)
