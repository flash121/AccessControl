from logistic_regression import OneHotEncoder, group_data, cv_loop

from copy import deepcopy
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
SEED = 25

def trainGeneralModel(all_data, num_train, y):
    
    # Transform data
    print "Transforming data..."
    dp = group_data(all_data, degree=2) 
    dt = group_data(all_data, degree=3)

    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]

    X_train_all = np.hstack((X, X_2, X_3))
    num_features = X_train_all.shape[1]
    
    model = linear_model.LogisticRegression()
    good_features = set([0, 8, 9, 10, 12, 19, 34, 36, 37, 38, 41, 42, 43, 47, 53, 60, 61, 63, 64, 67, 69, 71, 75, 81, 82, 85])
    good_features = sorted(list(good_features))
 
    print "Performing hyperparameter selection..."
    # Hyperparameter selection loop
    Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]
    score_hist = []
    N = 10
    Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
    Cvals = np.logspace(-2, 2, 7, base=2)
    for C in Cvals:
        model.C = C
        score = cv_loop(Xt, y, model, N)
        score_hist.append((score, C))
        
        print "C: %f Mean AUC: %f" %(C, score)
    bestC = sorted(score_hist)[-1][1]
    print "Best C value: %f" % (bestC)
    model.C = bestC
        
    return model, Xt
    

def evaluate(X, y, model, N):
    mean_auc = 0.
    mean_preds = np.zeros(len(y))
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X)[:,1]
        if all(y_cv == 1):
            continue
        auc = metrics.auc_score(y, preds)

        #print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_preds += np.array(preds)
        mean_auc += auc
    return mean_auc/N, mean_preds/N