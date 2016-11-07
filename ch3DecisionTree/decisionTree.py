# coding:utf-8
from math import log
import operator

def calcShannonEnt(dataSet):
    """
    计算熵
    :param dataSet: 输入数据,格式为二维数组,[['特征1','特征2','类别A'],[...],[...],[...]]
    :return: 熵值
    """
    numEntries = len(dataSet)  # 数据集的长度
    labelCounts = {}  # 类别标签键值对
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 一行数据的最后一个为类别
        if currentLabel not in labelCounts.keys():  # 如果键值对不存在则创建新对
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 计算每个类别的总数

    shannonEnt = 0  # 熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 每个类别的概率
        shannonEnt += -prob * log(prob, 2)  # 香农公式
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征划分数据集,如果dataset行元素的第axis个值等于value，
    则填充到特征值数组中
    :param dataSet: 输入数据
    :param axis: 划分特征
    :param value: 特征值
    :return: 特征的值
    """
    retDataSet = []
    for featVec in dataSet:  # 取每一个特征值
        if featVec[axis] == value:  # 抽取符合特征的元素
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
            print retDataSet
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return: 最佳特征的位置
    """
    numFeatures=len(dataSet[0])-1   # 特征的数量为单行元素的总数减去1,行末尾是类别值不是特征
    baseEntropy=calcShannonEnt(dataSet) # 整体数据集的熵
    bestInfoGain=0.0    #初始的最佳信息增益
    bestFeature=-1      #初始的最佳分类特征位置
    for i in range(numFeatures):    # 遍历每一个特征
        featList=[example[i] for example in dataSet]    # 抽取每个元素的第i个变量形成数组
        uniqueVals=set(featList)    # 合并重复项,创建分类标签列表
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet, i, value)  #拆分数据集为子数据集
            prob=len(subDataSet)/float(len(dataSet))    # 子数据集的概率
            newEntropy+=prob*calcShannonEnt(subDataSet) # 子数据集的熵
        infoGain=baseEntropy-newEntropy #计算划分之后的信息增益
        if infoGain>bestInfoGain:    #寻找最佳的划分特征和信息增益值
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    """
    返回出现次数最多的类别名称
    :param classList:
    :return:
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
            classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 输入数据集
    :param labels: 标签列表
    :return:
    """
    classList=[example[-1] for example in dataSet]  # 数据集最后一列为类别标签,把它添加到类别list中
    if classList.count(classList[0])==len(classList):#单结点的情况,类别完全相同, 停止划分
        return classList[0]
    if len(dataSet[0])==1:  #特征数为1的时候直接返回这个特征
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)  # 所有最佳的分类特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del labels[bestFeat]
    featValues=[example[bestFeat] for example in dataSet]   #得到列表包含的所有属性值
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat,value),subLabels)
    return myTree

def test():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [0, 1, 'no'], [1, 0, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    mytree=createTree(dataset,labels)

    #print calcShannonEnt(dataset)
    print splitDataSet(dataset,0,0)
    #print chooseBestFeatureToSplit(dataset)
    #print createTree(dataset,labels)
    pass


if __name__ == '__main__':
    test()
