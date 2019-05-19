"""
支持向量机类
    介绍：
        输入关节点信息（input_x），输出类别。每个关节点信息为一列表：[x1,x2, ...]
        input_x格式：[[x1_1, x1_2, ...], [x2_1, x2_2, ...], ... [xn_1, xn_2, ...]] #xi为数字
        input_y格式：[y1, y2, y3,...] #yi为数字或字符串
    初始化：   >>>mySvm = SVM(load = True) #load设为True表示加载之前训练过的参数继续训练
    训练：     >>>mySvm.train(input_x, input_y, save = True) #save设为True表示保存本次训练后的参数，注意会覆盖之前的数据
    测试：     >>>mySvm.test(input_x, input_y)
    应用：     >>>mySvm(input_x) #输出预测类别
"""
from sklearn import svm
import torch

class SVM():
    def __init__(self, load = True):
        self.savePath = "..\\weights\\"+SVM.__name__+".pkl"
        if not load:
            self.svm = svm.SVC(gamma='auto')
        else:
            try:
                with open(self.savePath , "rb") as f:
                    self.svm = torch.load(f)
            except FileNotFoundError:
                self.svm = svm.SVC(gamma='auto')

    def __call__(self, input):
        return self.svm.predict(input)

    def save(self):
        with open(self.savePath, "wb") as f:
            torch.save(self.svm, f)

    def train(self, X, Y, save = False):
        self.svm.fit(X,Y)
        if save:
            self.save()

    def test(self, X, Y):
        correct = 0
        sum = len(Y)
        result = self(X)
        for i,j in zip(Y, result):
            if i == j:
                correct+=1
        return correct/sum

def demo(): #使用示例
    mySVM = SVM(load=False)
    input_x = [[0,0],[1,1]]
    input_y = ['yes','no']
    mySVM.train(input_x, input_y, save = False)
    print("训练结束")
    input_x = [[0,-1], [1,3]]
    target_y = ['yes','no']
    p = mySVM.test(input_x, target_y)
    print("准确率：{0}%".format(100*p))

if __name__ == "__main__":
    demo()