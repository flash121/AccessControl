from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
f = open('result', 'w')
f.write('id,ACTION\n')

high = 0.9
low = 0.2

result = pd.read_csv("logistic_regression_pred.csv")
resultArray = result.values
for i in range(0, len(result)):
    if resultArray[i, 1] > high:
        resultArray[i, 1] = 1
    elif resultArray[i, 1] < low:
        resultArray[i, 1] = 0
    f.write(str(int(resultArray[i, 0])) + "," + str(resultArray[i, 1]) + "\n")


f.close()
        