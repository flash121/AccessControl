#import pdb
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
import trainModel
SEED = 25

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

def OneHotEncoder(data, keymap=None):
     """
     OneHotEncoder takes data matrix with categorical columns and
     converts it to a sparse binary matrix.
     
     Returns sparse binary matrix and keymap mapping categories to indicies.
     If a keymap is supplied on input it will be used instead of creating one
     and any categories appearing in the data that are not in the keymap are
     ignored
     """
     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i in prediction.keys():
        content.append('%i,%f' %(i,prediction[i]))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

# This loop essentially from Paul's starter code
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        if all(y_cv == 1):
            continue
        auc = metrics.auc_score(y_cv, preds)

        #print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N


    
def main(train='train.csv', test='test.csv', submit='logistic_pred.csv'):    
    print "Reading dataset..."
    #pdb.set_trace()
    train_data = pd.read_csv(train)

    test_data = pd.read_csv(test)
    all_data = np.vstack((train_data.ix[:,1:-1], test_data.ix[:,1:-1]))
    all_data_full = all_data
    
    # select a column to classify 
    col = 3
    # used later
    c_column = all_data[:,col]
    #cSet = set(c_column)
    #cSet = np.array([118213,118080,118041,118257,118170,119091,118888,118463,118291,117969,117903,118026,118446,118413,117962,118052,118386,118225,118327,118343,118300])
    #118080,118041,118257,118170,119091,118888,
    #cSet = np.array([118026, 118413, 118052, 118386, 118225, 118343, 118300, 118257, 118170, 118291, 117969])
    cSet = np.array([118041])
    temp_data = np.delete(all_data, col, axis = 1)
    all_data = temp_data

    num_train = np.shape(train_data)[0]
    
    # Transform data
    print "Transforming data..."
    dp = group_data(all_data, degree=2) 
    dt = group_data(all_data, degree=3)

    y = array(train_data.ACTION)
    
    X = all_data[:num_train]
    X_2 = dp[:num_train]
    X_3 = dt[:num_train]
    c_column_train = c_column[:num_train]
    
    X_test = all_data[num_train:]
    X_test_2 = dp[num_train:]
    X_test_3 = dt[num_train:]
    c_column_test = c_column[num_train:]
    
    X_train_all = np.hstack((X, X_2, X_3))
    X_test_all = np.hstack((X_test, X_test_2, X_test_3))
    num_features = X_train_all.shape[1]
    print num_features
    
    model_all, X_all = trainModel.trainGeneralModel(all_data_full, num_train, y)
    #generalPreds =[]
    N = 10
    allAUC, generalPreds = trainModel.evaluate(X_all, y, model_all, N)
    totaldict = {}
    geneScores = []
    fScores = []
    dScores = []
    for dID in cSet:
        
        print dID

        # select its corresponding resources
        print X_train_all.shape
        X_train = X_train_all[c_column_train == dID]
        y_train = y[c_column_train == dID]        
        
        print X_train.shape
        X_test = X_test_all[c_column_test == dID]
        #print np.array(test_data.id)
        testID = test_data[c_column_test == dID].id
        temp = np.array(testID)
        testID = temp
        
        # AUC
        gPreds = generalPreds[c_column_train == dID]
        #generalAUC = 0

        generalAUC = metrics.auc_score(y_train, gPreds)
        
        print generalAUC
        
        geneScores.append(generalAUC)
        
        model = linear_model.LogisticRegression()
        
        # Xts holds one hot encodings for each individual feature in memory
        # speeding up feature selection 
        Xts = [OneHotEncoder(X_train[:,[i]])[0] for i in range(num_features)]
        
        print "Performing greedy feature selection..."
        score_hist = []
        good_features = set([])
        # Greedy feature selection loop
        while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
            scores = []
            for f in range(len(Xts)):
                if f not in good_features:
                    feats = list(good_features) + [f]
                    Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
                    
                    #print Xt.shape
                    
                    score = cv_loop(Xt, y_train, model, N)
                    scores.append((score, f))
                    print "Feature: %i Mean AUC: %f" % (f, score)
    
            good_features.add(sorted(scores)[-1][1])
            score_hist.append(sorted(scores)[-1])
            print "Current features: %s" % sorted(list(good_features))
            
            
        # Remove last added feature from good_features
        good_features.remove(score_hist[-1][1])
        good_features = sorted(list(good_features))
        print "Selected features %s" % good_features
        
        print "Performing hyperparameter selection..."
        # Hyperparameter selection loop
        score_hist = []
        Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
        Cvals = np.logspace(-4, 4, 15, base=2)
        for C in Cvals:
            model.C = C
            score = cv_loop(Xt, y_train, model, N)
            score_hist.append((score,C))
            print "C: %f Mean AUC: %f" %(C, score)
        bestC = sorted(score_hist)[-1][1]
        bestAUC = sorted(score_hist)[-1][0]
        dScores.append(bestAUC)
        
        print "Best C value: %f, %f" % (bestC, bestAUC)
                
        model.C = bestC       
        fAUC, fPreds = trainModel.evaluate(Xt, y_train, model, N)

        fScores.append(fAUC)
        
        print "Performing One Hot Encoding on entire dataset..."
        Xt = np.vstack((X_train[:,good_features], X_test[:,good_features]))
        
        Xt, keymap = OneHotEncoder(Xt)
        num_train = np.shape(X_train)[0]
        X_train = Xt[:num_train]
        X_test = Xt[num_train:]
        
        print "Training full model..."
        model.fit(X_train, y_train)
        
        print "Making prediction and saving results..."
        pred = model.predict_proba(X_test)[:,1]

        totaldict.update(dict(zip(testID, pred)))
    print "group Set: ", cSet
    print "general scores: ", geneScores
    print 'seperate scores: ', dScores
    print 'full scores(corresponding to general scores): ', fScores 
    print len(totaldict)

    create_test_submission(submit, totaldict)
    
if __name__ == "__main__":
    args = { 'train':  'train.csv',
             'test':   'test.csv',
             'submit': 'logistic_regression_pred.csv' }
    main(**args)
    
