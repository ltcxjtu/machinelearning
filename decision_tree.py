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
            categories[a[-1]]=0
        else:
            categories[a[-1]]+=1

    shannonEnt=0
    for i in categories:
        shannonEnt+=-categories[i]/countSum*math.log(categories[i]/countSum,2)

    return shannonEnt