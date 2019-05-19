## 一、文件目录说明：
#### ./data     :   存放数据和数据集类和数据配置文件
#### ./models   :   存放各个模型类
#### ./others   :   杂七杂八，无工程用处
#### ./papers   :   论文
#### ./utils    :   实用的工具函数，例如实际应用中传入视频
#### ./weights  :   存储模型数据，下次训练时或实际应用时可以直接加载
#### main.py    :   主函数文件，用于最终的实际【运行】
#### test.py    :   测试文件，用于【运行】
#### train.py   :   训练文件，用于【运行】

## 二、我们的任务
#### step1[hard] : 实现该工程的三部分主要模型 Faster R-CNN\SHDL\SVM.
#### step2[easy] : 训练和测试 终端下运行 python train.py <mode> \ python test.py <mode>
#### step3[easy] : 实际应用 终端下运行 python main.py
#### step4[hard] : 优化 改写成GPU版本进行深度训练 用C++版本的pytorch调用训练后的参数