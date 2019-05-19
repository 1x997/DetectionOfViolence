"""
测试文件
"""
import sys

if __name__ == "__main__":
    #测试模式
    if sys.argv[1] == 'FRCNN':
        pass
    elif sys.argv[1] == 'SHDL':
        pass
    elif sys.argv[1] == 'SVM':
        pass
    else:
        print('请输入合法参数：FPCNN or SHDL or SVM')