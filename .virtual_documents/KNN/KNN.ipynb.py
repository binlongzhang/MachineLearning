from numpy import *
import operator
import matplotlib.pyplot as plt

# ��������
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
group,labels = createDataSet()

plt.scatter(group[:,0],group[:,1])
# ���ϱ�ǩ
for i in range(len(labels)):
    plt.annotate(labels[i], xy = (group[i][0], group[i][1]), xytext = (group[i][0]+0.01, group[i][1]+0.01))

plt.show()


# ����KNN��ʹ��ŷ�Ͼ���
def classify0(inputX, dataSet, labels, k):
    ## inputX��һ����������  dataSet��ѵ��������   lebels: dataSet�ı�ǩ  K��KNN��k������
    dataSetSize = dataSet.shape[0]
    
    # diffMat�������������������ݼ��Ĳ�࣬tile ��ԭʼ�����������Ķѵ�
    diffMat = tile(inputX, (dataSetSize,1)) - dataSet
    
    # ����������ŷ�Ͼ���
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    # ��λ�������������ļ�������
    # �������򷵻�����
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # dict.get(key, default=None)�����򷵻ض�Ӧ�ļ�ֵ�����򷵻�0
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    # k�������򣬲����ر�ǩ    
    sortedClassCount = sorted(classCount.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
    return sortedClassCount[0][0]


classify0([0,0],group,labels,3)


classify0([1,1],group,labels,3)


classify0([0.5,0.5],group,labels,3)


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)

    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # ɾ����λ�ո�
        line = line.strip()
        # ��\tΪ�ָ����ָ��ַ���
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        
        # ��ӱ�ǩ
        # if listFromLine[-1] == 'didntLike':
        #     classLabelVector.append(1)
        # elif listFromLine[-1] == 'smallDoses':
        #     classLabelVector.append(2)
        # elif listFromLine[-1] == 'largeDoses':
        #     classLabelVector.append(3)
        classLabelVector.append(int(listFromLine[-1]))

        index += 1
    return returnMat,classLabelVector



filename = '../data/KNN/datingTestSet.txt'
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
    datingDataMat,datingLabels = file2matrix('../data/KNN/datingTestSet.txt')
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


filename = '../data/KNN/datingTestSet.txt'
classifyPerson(filename)


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect        


filename = '../data/KNN/TrainingDigits/0_13.txt'
testVector = img2vector(filename)
testVector[0,0:31]


from os import listdir

def handWritingClassTest(trainDir,testDir):
    hwLabels = []
    trainingFileList = listdir(trainDir)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    # ��ȡѵ������
    for i in range(m):
        # ��ȡ��ǩ
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # ��ȡ����
        trainingMat[i,:] = img2vector(trainDir + '/' + fileNameStr)

    testFileList = listdir(testDir)
    errorCount = 0
    mTest = len(testFileList)
    
    for i in range(mTest):
        # ��ȡ��ǩ
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # ��ȡ���ݾ�
        vectorUnderTest = img2vector(testDir + '/' + fileNameStr)
        # ����
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        # ͳ��
        if (classifierResult get_ipython().getoutput("= classNumStr):")
            errorCount +=1
    # ��ӡ���
    print('the total number of error is :get_ipython().run_line_magic("d'", " % errorCount)")
    print('the total error rate is : get_ipython().run_line_magic("f'", " % (errorCount/float(mTest)))")



trainDir = '../data/KNN/trainingDigits'
testDir = '../data/KNN/testDigits'
handWritingClassTest(trainDir,testDir)
