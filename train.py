"""
训练文件
终端下运行：python train.py <mode>
其中mode有FRCNN SHDL SVM 三种，表示分别训练对应的模型
"""
import sys

if __name__ == "__main__":
    #训练模式
    if sys.argv[1] == 'FRCNN':
        pass
    elif sys.argv[1] == 'SVM':
        pass
    else:
        print('请输入合法参数：FPCNN or SVM')
