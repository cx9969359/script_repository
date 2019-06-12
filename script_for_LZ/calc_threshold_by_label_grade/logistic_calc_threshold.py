import argparse

import yaml
from numpy import *


def scan_pickle_sub_folder(pickle_file_directory):
    path = os.path.join(pickle_file_directory)
    if not os.path.isdir(path):
        raise Exception('No such pickle_file_directory')
    return os.listdir(pickle_file_directory)


def scan_pickle_file(directory_path):
    path = os.path.join(directory_path)
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1].lower() == 'pkl':
                file_list.append(file)
    return file_list


def calc_annotation_num(pickle_file_path, label):
    with open(pickle_file_path, 'rb') as f:
        result = pickle.load(f)
        return len(result.get(label, []))


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('yml_path', type=str, help='path to pkl_files')
    args = parser.parse_args()
    yml_file = open(args.yml_path, encoding='utf-8')
    param_dict = yaml.safe_load(yml_file)
    for item in param_dict:
        parser.add_argument(item, type=type(param_dict[item]), default=param_dict[item])
    args = parser.parse_args()
    return args


def loadDataSet():
    dataMat, labelMat = [], []
    fr = open('F:/script_for_LZ/calc_threshold_by_label_grade/logistic_test_data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 特征数据集，添加1是构造常数项x0
        labelMat.append(int(lineArr[-1]))  # 分类数据集
    return dataMat, labelMat


def sigmoid(inX):
    return 1 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # (m,n)
    labelMat = mat(classLabels).transpose()  # 转置后(m,1)
    m, n = shape(dataMatrix)
    weights = ones((n, 1))  # 初始化回归系数，(n,1)
    alpha = 0.001  # 定义步长
    maxCycles = 500  # 定义最大循环次数
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # sigmoid 函数
        error = labelMat - h  # 即y-h，（m,1）
        weights = weights + alpha * dataMatrix.transpose() * error  # 梯度上升法
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    n = shape(dataMat)[0]
    xcord1 = []  
    ycord1 = []
    xcord2 = []  
    ycord2 = []
    for i in range(n):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3, 3, 0.1)
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]  # matix
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    # args = parse_arg()
    # root_directory = args.pkl_directory
    # grade_label_list = args.grade_label_list
    # highest_sensitivity = args.highest_sensitivity
    # sub_folder_list = scan_pickle_sub_folder(root_directory)
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotBestFit(weights)
