# coding:utf-8

from math import log
import operator


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shanninEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shanninEnt += -prob * log(prob, 2)
    return shanninEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            redecedFeatVec = featVec[:axis]
            redecedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(redecedFeatVec)
    return retDataSet


def chooseBestFeatureToSeplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    baseInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > baseInfoGain:
            baseEntropy = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSeplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = \
            createTree(splitDataSet(dataSet, bestFeat, value),
                       subLabels)
    return myTree


def test():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 0, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    myTree = createTree(dataset, labels)
    print myTree


# ---------plto tree----------
import matplotlib.pyplot as plt

decisionNode = dict(boxstype="sawtooth", fc="0.8")  # 定义判断节点形态
leafNode = dict(boxstyle="round4", fc="0.8")  # 定义叶节点形态
arrow_args = dict(arrowstyle="<-")  # 定义箭头


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制带箭头的注解
    :param nodeTxt: 节点的文字标注
    :param centerPt: 节点中心位置
    :param parentPt: 箭头起点位置（上一节点位置）
    :param nodeType: 节点属性
    :return:
    """
    createPlot.ax.annotate(nodeTxt,
                           xy=parentPt,
                           xycoords='ax fraction',
                           xytext=centerPt,
                           textcoords='axes fraction',
                           va='center',
                           ha='center',
                           bbox=nodeType,
                           arrowprops=arrow_args)


# 创建绘图区域
def createPlot():
    fig = plt.figure(1, facecolor="gray")  # 定义一个绘图区域
    fig.clf()
    createPlot.ax = plt.subplot(111, frameon=False)
    plotNode(u'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(u'叶子节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


if __name__ == '__main__':
    # test()
    createPlot()
