# coding:utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([1.0, 1.1],
                  [1.0, 1.0],
                  [0, 0],
                  [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    """
    knn算法核心
    :param inX:待分类的输入特征向量
    :param dataSet: 训练集
    :param labels: 标签集
    :param k: 参数k
    :return: 输入向量的类别标签
    """
    dataSetSize = dataSet.shape[0]  # 训练集行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 计算样本与训练集的差值矩阵,tile将输入向量转换成4＊2的矩阵
    sqDiffMat = diffMat ** 2  # 差值矩阵的平方
    sqDistance = sqDiffMat.sum(axis=1)  # 行元素相加求和,axis=0代表普通的相加
    distance = sqDistance ** 0.5  # 每一行元素的最终距离
    sortedDistanceIndicies = distance.argsort()  # 按照距离排序
    classCount = {}  # 距离最小点的集合
    for i in range(k):
        voteILabel = labels[sortedDistanceIndicies[i]]  # 获取前k个元素的标签
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1  # 计算标签出现的次数
        sortedClassCount = sorted(classCount.iteritems(),
                                  key=operator.itemgetter(1),
                                  reverse=True)  # 建立"标签--次数"对
    return sortedClassCount[0][0]  # 返回最相似的标签


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  # 文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建"行数*3"的全零矩阵
    classLabelVector = []  # 类别标签
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        print listFromLine[0:3]
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat, classLabelVector


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect


from os import listdir


def handwritingClassTest():
    hwLabels = []
    traingintFileList = listdir('./data/trainingDigits')
    m = len(traingintFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = traingintFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./data/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('./data/testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./data/testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount / float(mTest))


if __name__ == '__main__':
    # 分类问题
    # dataset = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    # labels = ['A', 'A', 'B', 'B']
    # x = array([1.2, 1.1])
    # y = array([0.1, 0.1])
    # k = 3
    # labelX = classify(x, dataset, labels, k)
    # labelY = classify(y, dataset, labels, k)
    # print labelX + '--'+ labelY

    # 错误率问题
    # datingDataMat, datingLabels = file2matrix('./data/datingTestSet.txt')
    # print datingLabels[:20]

    # 数字识别问题
    handwritingClassTest()
