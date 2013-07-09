import numpy as np
import pandas as pd

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
    index = (resArray == idRes)
    data = content[index]
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
    shapePath = path.shape
    p = np.zeros(shapePath)
    for i in range(len(path)):
        p[i] = np.where( np.array(path[i : i + 1]) == arrSinglePath, 1 , 0)
    
    arrAccess = np.asarray(access)
    pT = np.vstack((np.asarray(access), p.T))
    # now p is transposed
    pT = np.delete(pT, (1), axis = 0)
    
    X = []
    for i in range(len(pT)):

        s = arrAccess[pT[i] == 1]
        if s is None:
            X.append(0.5)
            continue
        X.append(np.mean(s))
    return X
        

        
content = readCSV('train.csv')
access, path = resource(0, content)
X = compare(path[:1], 0, content)