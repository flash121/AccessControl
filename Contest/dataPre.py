import numpy as np
import pandas as pd

# read in csv and return header and content seperately
def readCSV(filename):
    content = pd.read_csv(filename)
    print content
    return content

def resource(idRes, content):
    resource = content.RESOURCE
    resArray = np.array(resource)
    index = (resArray == idRes)
    data = content[index]
    #print data
    access = data.ACTION
    path = data.drop('ACTION', axis = 1)
    #print path
    return access, path

def compare(singlePath, idRes, content):
    arrSinglePath = np.asarray(singlePath)
    access, path = resource(idRes,content);
    pLen = len(path)
    shapePath = path.shape
    p = np.zeros(shapePath)
    for i in range(pLen):
        p[i] = np.where( np.array(path[i : i + 1]) == arrSinglePath, 1 , 0)
    arrAccess = np.asarray(access)

    arrAccess = np.transpose(arrAccess)
    p = np.vstack((np.asarray(access), p.T))
    # now p is transposed
    p = np.delete(p, (1), axis = 0)
    
    rlen= len(p)
    X = []
    for i in range(rlen):

        s = arrAccess[p[i] == 1]
        if s is None:
            X.append(0.5)
            continue
        X.append(np.mean(s))
    return X
        
        
content = readCSV('train.csv')
access, path = resource(0, content)
print path[:1]
X = compare(path[:1], 0, content)