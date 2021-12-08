def createDataSet():
    
    dataSet = [[1,1,'yes'],[1,1,'yes'],\
              [1,0,'no'],[0,1,'no'],[0,1,'no'] ]
    labels = ['no surfacing','flippers']
    
    return dataSet,labels

myDat,labels = createDataSet()
myDat


# 计算香农熵
from math import log
def calcShannonEnt(dataSet):
    numEnt = len(dataSet)
    labelCounts = {}
    # 统计数据
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 计算信息熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEnt
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

calcShannonEnt(myDat)


myDat[0][-1] = 'no'
calcShannonEnt(myDat)


myDat[0][-1] = 'maybe'
calcShannonEnt(myDat)


def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

myDat,labels = createDataSet()                                  
myDat


splitDataSet(myDat,0,1)


splitDataSet(myDat,0,0)


# 基于信息增益选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEnt = calcShannonEnt(dataSet)
    dataSetNum = len(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEnt = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(dataSetNum)
            newEnt += prob * calcShannonEnt(subDataSet)

        infoGain = baseEnt - newEnt
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        return bestFeature
    
chooseBestFeatureToSplit(myDat)


import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]

    # 叶节点样本都属于统一类别的时候返回
    setClassList = set(classList)
    if len(setClassList)==1:
        return classList.pop()

    # 遍历玩所有特征时候返回出现次数最多的类标签
    if len(dataSet[0])==1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatBabel = labels[bestFeat]

    myTree = {bestFeatBabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatBabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


myDat,labels = createDataSet()
myTree = createTree(myDat,labels)
myTree


import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = 'sawtooth',fc = '0.8')
leafNode = dict(boxstyle = 'round4',fc='0.8')
arrow_args = dict(arrowstyle = '<-')

def plotNode(subfig,nodeTxt,centerPt,parentPt,nodeType):
    subfig.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
                           xytext=centerPt,textcoords='axes fraction',\
                           va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)
    
    
def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    subfig = plt.subplot(111,frameon = False)
    plotNode(subfig,'a decision Node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(subfig,'a leaf Node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
createPlot()


# 获取叶节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            depth = 1 + getTreeDepth(secondDict[key])
        else:
            depth = 1
    if maxDepth>depth:
        return maxDepth
    else:
        return depth


def retrieveTree(i):
    listOfTrees =[\
                {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},\
                {'no surfacing': {0: 'no', 1: \
                      {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}\
                  ]
    return listOfTrees[i]
myTree = retrieveTree(0)
print(myTree)
print(getNumLeafs(myTree))
print(getTreeDepth(myTree))


# 在父子节点之间填充文本信息
def plotMidText(cntrPt,parentPt,txtString,TreeAtt):
    xMid = (parentPt[0]-cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2 + cntrPt[1]
    TreeAtt['fig'].text(xMid,yMid,txtString)
    
def plotTree(myTree,parentPt,nodeTxt,TreeAtt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    
    # 绘制选择节点
    firstStr = list(myTree.keys())[0]
    cntrPt = (TreeAtt['xOff']+(1.0 + float(numLeafs))/2/TreeAtt['totalW'] , TreeAtt['yOff'])
    plotMidText(cntrPt, parentPt, nodeTxt,TreeAtt)
    plotNode(TreeAtt['fig'],firstStr, cntrPt, parentPt, decisionNode)
    
    secondDict = myTree[firstStr]
    TreeAtt['yOff'] = TreeAtt['yOff'] - 1.0/TreeAtt['totalD']
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            # 依据根绘制树
            plotTree(secondDict[key],cntrPt,str(key),TreeAtt)
        else:
            # 绘制属性线
            TreeAtt['xOff'] = TreeAtt['xOff'] + 1.0/TreeAtt['totalW']
            plotNode(TreeAtt['fig'],secondDict[key], (TreeAtt['xOff'], TreeAtt['yOff']), cntrPt, leafNode)
            plotMidText((TreeAtt['xOff'], TreeAtt['yOff']), cntrPt, str(key),TreeAtt)
    
    # 回溯完了之后恢复y轴绘图坐标
    TreeAtt['yOff'] = TreeAtt['yOff'] + 1.0/TreeAtt['totalD']
    
def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    # 清除数据     
    fig.clf()
    TreeAtt = {}
    TreeAtt['fig'] = plt.subplot(111,frameon=False,xticks = [],yticks = [])
    TreeAtt['totalW'] = float(getNumLeafs(inTree))
    TreeAtt['totalD'] = float(getTreeDepth(inTree))
    TreeAtt['xOff'] = -0.5 / TreeAtt['totalW']
    TreeAtt['yOff'] = 1.0
    
    plotTree(inTree,(0.5,1.0),'',TreeAtt)
    plt.show()
    
createPlot(myTree)


myTree['no surfacing'][3] = 'maybe'
myTree


createPlot(myTree)


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if isinstance(secondDict[key],dict):
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


myDat,labels = createDataSet()
myTree = retrieveTree(0)
myTree


labels


print(classify(myTree,labels,[1,0]))
print(classify(myTree,labels,[1,1]))


# 使用pickle序列化对象
import pickle
def storeTree(inputTree,filename):
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

storeTree(myTree,'../data/DecTree/classifierStorage.txt')



grabTree('../data/DecTree/classifierStorage.txt')


fr = open('../data/DecTree/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
lensesTree


createPlot(lensesTree)



