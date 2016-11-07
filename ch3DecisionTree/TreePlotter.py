# coding:utf-8
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
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords='ax fraction',
                            xytext=centerPt,
                            textcoords='axes fraction',
                            va='center',
                            ha='center',
                            bbox=nodeType,
                            arrowprops=arrow_args)


def getNumLeafs(myTree):
    """
    计算叶子节点的数目
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 不是字典的数据就是叶子节点
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1  # 如果是叶节点,则叶节点+1
    return numLeafs


def getTreeDepth(myTree):
    """
    计算叶子节点的层数,即树的深度
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1

        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    """
    在父子节点间填充文本信息
    :param cntrPt:子节点位置
    :param parentPt:父节点位置
    :param txtString:标注内容
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[0]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    """
    递归绘制决策树
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 创建绘图区域
def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")  # 定义一个绘图区域
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0:{'flippers': {0: 'no', 1: 'yes'}}, 1: {'flippers': {0: 'no', 1: 'yes'}}, 2:{'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


if __name__ == '__main__':

    createPlot(retrieveTree(0))
    # http://www.cnblogs.com/fantasy01/p/4595902.html