def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)

    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # 删除首位空格
        line = line.strip()
        # 以\t为分隔符分割字符串
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        
        # 添加标签
        # if listFromLine[-1] == 'didntLike':
        #     classLabelVector.append(1)
        # elif listFromLine[-1] == 'smallDoses':
        #     classLabelVector.append(2)
        # elif listFromLine[-1] == 'largeDoses':
        #     classLabelVector.append(3)
        classLabelVector.append(int(listFromLine[-1]))

        index += 1
    return returnMat,classLabelVector



filename = './data/KNN/datingTestSet.txt'
datingDatMat,datingLabels = file2matrix(filename)
datingDatMat


datingLabels[0:20]


import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDatMat[:,1],datingDatMat[:,2],15*array(datingLabels),15*array(datingLabels))
plt.show()


def autoNorm(dataSet):
    
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    
    return normDataSet, ranges, minVals

normMat,ranges,minvals = autoNorm(datingDatMat)
normMat


ranges


minvals


def datingClassTest():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('data/KNN/datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDatMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        # print('the classifier come back with: get_ipython().run_line_magic("d,", " the real answer is : %d' % (classifierResult,datingLabels[i]))")
        if (classifierResult get_ipython().getoutput("= datingLabels[i]):")
            errorCount += 1
    print('the total error rate is : get_ipython().run_line_magic("f'%(errorCount/float(numTestVecs)))", "    ")


datingClassTest()


def classifyPerson(filename):
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))

    datingDataMat,datingLabels = file2matrix(filename)
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inputArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0( (inputArr-minVals)/ranges, normMat,datingLabels, 3)
    print('You will probably like this person: ',resultList[classifierResult-1])


filename = 'data/KNN/datingTestSet.txt'
classifyPerson(filename)
