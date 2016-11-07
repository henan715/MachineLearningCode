# coding:utf-8

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    dataMat = []  # 特征
    labelMat = []  # 类别
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    """
    拟合最佳直线
    :param xArr: 输入特征向量
    :param yArr: 特征向量的类别
    :return: 回归系数矩阵
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # det用于矩阵运算
        print '输入矩阵是奇异的,无法求解.'
        return
    ws = xTx * (xMat.T * yMat)  # 回归系数
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    局部线性加权
    :param testPoint:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))  # 创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print 'matrix is singular'
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def test():
    xArr, yArr = loadDataSet('./data/ex0.txt')
    j = 1
    yHat = lwlrTest(xArr, xArr, yArr, 1)
    yHat2 = lwlrTest(xArr, xArr, yArr, 0.1)
    yHat3 = lwlrTest(xArr, xArr, yArr, 0.03)
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')

    ax2 = fig.add_subplot(312)
    ax2.plot(xSort[:, 1], yHat2[srtInd])
    ax2.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')

    ax3 = fig.add_subplot(313)
    ax3.plot(xSort[:, 1], yHat3[srtInd])
    ax3.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')

    plt.show()


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


def abaloneTest():
    abX, abY = loadDataSet('./data/abalone.txt')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1  = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print rssError(abY[0:99], yHat01.T)
    print rssError(abY[0:99], yHat1.T)
    print rssError(abY[0:99], yHat10.T)

def ridgeRegres(xMat, yMat, lam=0.2):
    """
    岭回归
    :param xMat:
    :param yMat:
    :param lam: 调节系数
    :return:
    """
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:  # lam设置为0的时候还有可能导致矩阵奇异
        print '矩阵奇异,不可计算'
        return
    ws=denom.I*(xMat.T*yMat)
    return ws

def ridgeTest(xArr, yArr):
    """
    利用岭回归和缩减技术,首先要对特征作标准化处理(值减去均值,再除以方差)
    :param xArr:
    :param yArr:
    :return:
    """
    xMat=mat(xArr)
    yMat=mat(yArr).T
    yMean=mean(yMat,0)  # 均值
    yMat=yMat-yMean     # 减去均值
    xMeans=mean(xMat,0) # 均值
    xVar=var(xMat,0)    # 方差
    xMat=(xMat-xMeans)/xVar # x-均值,再除以方差,实现标准化
    numTestPts=30
    wMat=zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def abaloneTestWithRidge():
    abX, abY = loadDataSet('./data/abalone.txt')
    ridgeWeights=ridgeTest(abX,abY)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

if __name__ == '__main__':
    # test()
    #abaloneTest()
    abaloneTestWithRidge()
