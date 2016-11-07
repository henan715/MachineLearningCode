# coding:utf-8

from numpy import *
import numpy as np
import operator
from matplotlib import pyplot as plt


def createDataSet():
    datas = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return datas, labels


def classify(inX, dataSet, labels, k):
    """
    KNN Algorithm
    :param inX: input feature vector
    :param dataSet: training data
    :param labels: labelset
    :param k: parameter k
    :return: class labels of input
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistanceIndicies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistanceIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line=line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    归一化数据源,公式为:(x-min)/(max-min)
    :param dataSet:
    :return:
    """
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.1
    datingData, datingLabel = file2matrix('./data/datingTestSet.txt')
    normMat,ranges,minValus=autoNorm(datingData)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify(normMat[i,:],normMat[numTestVecs:m,:],
                                  datingLabel[numTestVecs:m],3)
        print "预测: %d, 实际: %d" % (classifierResult, datingLabel[i])
        if classifierResult!=datingLabel[i]:
            errorCount+=1.0
    print "错误率:%f"%(errorCount/float(numTestVecs))

# --------------image-----------
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

import os
def handWritingClassTest():
    hwLabels=[]
    trainingFileList=os.listdir('./data/trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('./data/trainingDigits/%s'%fileNameStr)
    testFileList=os.listdir('./data/testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('./data/testDigits/%s'%fileNameStr)
        classfierResult=classify(vectorUnderTest, trainingMat,hwLabels,3)
        print "预测: %d, 实际: %d" % (classfierResult, classNumStr)
        if classfierResult!=classNumStr:
            errorCount+=1.0
    print '错误数目:',errorCount
    print "错误率:%f"%(errorCount/float(mTest))

from sklearn import neighbors
def handWritingClassTestWithSklearn():
    # 用sklearn的KNN模块进行预测
    knn = neighbors.KNeighborsClassifier()

    hwLabels = []
    trainingFileList = os.listdir('./ch2KNN/data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./ch2KNN/data/trainingDigits/%s' % fileNameStr)
    knn.fit(trainingMat,hwLabels)
    testFileList = os.listdir('./ch2KNN/data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./ch2KNN/data/testDigits/%s' % fileNameStr)
        #classfierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        classfierResult = knn.predict(vectorUnderTest)
        print "预测: %d, 实际: %d" % (classfierResult, classNumStr)
        if classfierResult != classNumStr:
            errorCount += 1.0
    print '错误数目:', errorCount
    print "错误率:%f" % (errorCount / float(mTest))

if __name__ == '__main__':
    # datas, labels = createDataSet()
    # print classify([0, 0], datas, labels, 3)

    datingData, datingLabel = file2matrix('./data/datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingData[:, 0], datingData[:, 1],
              15.0 * array(datingLabel),
              15.0 * array(datingLabel))
    plt.xlabel('play games time')
    plt.ylabel('icerame consumption')
    plt.show()

    #datingClassTest()

    #handWritingClassTest()
