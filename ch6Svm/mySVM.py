# coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJRand(i, m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj, h, l):
    if aj>h:
        aj=h
    if l>aj:
        aj=l
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix=mat(dataMatIn)
    labelMatrix=mat(classLabels).transpose()
    b=0
    m,n=shape(dataMatrix)
    alphas=mat(zeros(m,1))
    iter=0
    while(iter<maxIter):
        alphaPairsChanged=0
        for i in range(m):
            fXi=float(multiply(alphas, labelMatrix).T*\
                      (dataMatrix*dataMatrix[i,:].T))+b
            Ei=fXi-float(labelMatrix[i])
            if ((labelMatrix[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMatrix[i]*Ei>toler) and (alphas[i]>0)):
                j=selectJRand(i,m)
                fXj=float(multiply(alphas,))



def test(i):
    if i==1:
        dataArr,labelArr=loadDataSet('./data/testSet.txt')
        print labelArr[:20]


if __name__=='__main__':
    test(1)