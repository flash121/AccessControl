import numpy as np
import pandas as pd
import random
# read in csv and return header and content seperately

def readCSV(filename):
    content = pd.read_csv(filename)
    print content
    return content


# idRes is an resource id we want to find its info
# content is the loaded file as dataframe format
def resource(idRes, content):
    resource = content.RESOURCE
    resArray = np.array(resource)
    #print idRes
    #print resArray
    data = content[resArray == idRes]
    #print 'data', data
    #print data
    access = data.ACTION
    path = data.drop('ACTION', axis = 1)
    #print path
    # access is the action(0/1)
    # path is all other info except action
    return access, path


# singlePath is one row of info for testing
# idRes is resource id
# content is the info
def compare(singlePath, idRes, content):


    arrSinglePath = np.asarray(singlePath)
    access, path = resource(idRes,content);
    #print access, path
    #print arrSinglePath


    shapePath = path.shape
    p = np.zeros(shapePath)
    for i in range(len(path)):
        #print np.array(path[i : i + 1])
        p[i] = np.where( np.array(path[i : i + 1]) == arrSinglePath, 1 , 0)
    #print p
    arrAccess = np.asarray(access)
    pT = np.vstack((np.asarray(access), p.T))
    # now p is transposed
    pT = np.delete(pT, (1), axis = 0)
    #print pT
    X = []
    for i in range(len(pT)):


        s = arrAccess[pT[i] == 1]
        #print 's', s, len(s)
        if len(s) == 0:
            X.append(0.5)
            continue
        X.append(np.mean(s))
    return X


def genRandom(n):
    rNum = [random.randint(1,10) for i in range(n)]
    return rNum


def selectSample(content, r, threshold):
    r = np.array(r)
    
    trainSet = content[r > threshold]
    testSet = content[r <= threshold]


    return trainSet, testSet


def genVector(trainSet, testSet):
    X = []
    for i in range(len(testSet)):
        onePath = testSet[i:i+1]
        #print onePath      
        idRes = onePath.RESOURCE
        idRes = np.array(idRes)
        res = idRes[0]
        
        Xtemp = compare(onePath, res, trainSet)
        X.append(Xtemp)
        #print i
        #print Xtemp
    return X

def getTest():
    testContent = readCSV('test.csv')
    content = readCSV('train.csv')

    testC = testContent.drop('id', axis = 1)

    X = genVector(content, testC)
    #print X
    return X

def create_test_submission(prediction):
    content = ['Id,Action']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open("submittion.csv", 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'
    
def getdata():            
    content = readCSV('train.csv')
    n = len(content)
    r = genRandom(n)
    trainSet, testSet = selectSample(content, r, 1)
    #print trainSet
    #print testSet

    testS = testSet.drop('ACTION', axis = 1)
     
    X = genVector(trainSet, testS)
    A2=testSet['ACTION']
    A=[]
    for a in range(0,A2.index.shape[0]):
        A.append(A2[A2.index[a]])
    return A,X

if __name__=='__main__':
    getTest()

