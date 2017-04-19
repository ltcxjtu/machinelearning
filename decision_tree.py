import math
def createDateSet():
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surdacing','flippers']
    return dataSet,labels

def calculateShannonEnt(dataSet):
    countSum=len(dataSet)
    categories={}
    for a in dataSet:
        if a[-1] not in categories.keys():
            categories[a[-1]]=1
        else:
            categories[a[-1]]+=1

    shannonEnt=0
    for i in categories:
        shannonEnt+=-categories[i]/countSum*math.log(categories[i]/countSum,2)

    return shannonEnt

def splitDataSet(dataSet,axis,value):
    reDataSet=[]
    for a in dataSet:
        if a[axis]==value:
            reData=a[:axis]
            reData.extend(a[axis+1:])
            reDataSet.append(reData)
    return reDataSet


def chooseBestFeature(dataSet):
    #计算每个特征的条件熵
    #1.划分数据集;2.数据集下有几个分类;3.计算条件熵;
    #4.计算信息增益
    numFeature=len(dataSet[0])-1
    shannonEnt=calculateShannonEnt(dataSet)
    infGain=0
    bestFeature=-1
    for i in range(numFeature):
        categories={}
        reDataSets=[]
        for a in dataSet:
            if a[i] not in categories.keys():
                categories[a[i]]=1
            else:
                categories[a[i]]+=1
        for b in categories:
            reDataSets.append(splitDataSet(dataSet,i,b))
        condShannonEnt=0
        for j,c in zip(categories.keys(),reDataSets):
            condShannonEnt+=categories[j]/len(dataSet)*calculateShannonEnt(c)

        infGaintemp=shannonEnt-condShannonEnt

        if infGain<infGaintemp:
            infGain=infGaintemp
            bestFeature=i

    return  bestFeature

def majorityCnt(dataSet):
    classCount={}
    for a in dataSet:
        if a[-1] not in classCount.keys():
            classCount[a[-1]]=1
        else:
            classCount[a[-1]]+=1
    sortclassCnt=sorted(classCount.items(),key=lambda a:a[1],reverse=True)
    return sortclassCnt[0][0]



if __name__=='__main__':
     dataSet,labels=createDateSet()
     shannonEnt=calculateShannonEnt(dataSet)
     reDataSet=splitDataSet(dataSet,0,1)
     bestFeature=chooseBestFeature(dataSet)
     c=majorityCnt(dataSet)
     print(reDataSet)
