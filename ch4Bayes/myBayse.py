# coding:utf-8

from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    处理训练集，看看训练集有多少个不同的（唯一的）单词组成
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 求两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    返回一个由唯一单词组成的词汇表
    :param vocabList: 词汇表
    :param inputSet: 输入文档
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 01代表词汇表中的单词在文档中是否出现
        else:
            print '==%s== is not in my vocabulary.' % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: 文档矩阵
    :param trainCategory: 每篇文档的标签组成的矩阵
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 获取总训练样本数目
    numWords = len(trainMatrix[0])  # 获取总单词数目
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 类别概率p(Ci),即侮辱性评论的概率,正常评论的概率用1减去该值即可
    p0Num = ones(numWords)  # 正常评论单词数目,P(Wi|C0)的分子
    p1Num = ones(numWords)  # 侮辱评论单词数目,P(Wi|C1)的分子
    p0Denom = 2.0  # 初始概率,P(Wi|C0)的分母
    p1Denom = 2.0  # 初始概率,P(Wi|C1)的分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 若某个词语在某个文档中出现,该词语对应的数目加一
            p1Denom += sum(trainMatrix[i])  # 在所有文档中,该文档的总词数加一
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vec = log(p1Num / p1Denom)  # 侮辱文档的概率
    p0Vec = log(p0Num / p0Denom)  # 正常文档的概率
    return p0Vec, p1Vec, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOposts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, "classified as:", classifyNB(thisDoc, p0v, p1v, pAb)

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, "classified as:", classifyNB(thisDoc, p0v, p1v, pAb)


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in vocabList:
        returnVec[vocabList.index(word)] += 1
    return returnVec

# -------email test------------

def textParse(bigString):
    """
    处理文本,根据正则表达式将文本切分,返回长度大于2的文本集合
    :param bigString:
    :return:
    """
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainingMat = []
    trainingClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p0V, p1V, pAb = trainNB0(array(trainingMat), array(trainingClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pAb) != classList[docIndex]:
            errorCount += 1
    print 'error rate is:', float(errorCount) / len(testSet)


if __name__ == "__main__":
    testingNB()
