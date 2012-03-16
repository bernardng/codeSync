"""
Classification of IMAGEN faces data
"""
import os
from scipy import io
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn import grid_search

BASE_DIR = "/volatile/bernardng/data/imagen/"
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/facesList.txt"), dtype='str')

thresh = 0.56
n_samp_trial = 8 # Manually determined
n_folds = 4
n_ifolds = 3

method = 4

if method == 0:
    # Logistic Regression
    classifier = LogisticRegression(penalty='l2', fit_intercept=False, scale_C=True)
    parameters = {'C':[1e16]}
elif method == 1:
    # SVM
    classifier = svm.LinearSVC(fit_intercept=False, scale_C=True)
#    parameters = {'C':np.logspace(0, 6, 20)}
    parameters = {'C':np.linspace(1e0, 1e6, 10)}
elif method == 2:
    # l2 regularized Logistic Regression
    classifier = LogisticRegression(penalty='l2', fit_intercept=False, scale_C=True)
#    parameters = {'C':np.logspace(0, 6, 20)}
    parameters = {'C':np.linspace(1e0, 1e6, 10)}
elif method == 3:
    # Sparse Logistic Regression
    classifier = LogisticRegression(penalty='l1', fit_intercept=False, scale_C=True)
#    parameters = {'C':np.logspace(0, 6, 30)}
    parameters = {'C':np.linspace(1e0, 1e6, 10)}
elif method == 4:
    # Connectivity-informed Logistic Regression
#    classifier = LogisticRegression(penalty='l2', C=1e16, fit_intercept=False, scale_C=True)
    classifier = LogisticRegression(penalty='l2', C=1e16, fit_intercept=False, scale_C=True)
    parameters = np.linspace(1e0, 1e6, 10)
    

for sub in [subList[0]]:
    print str("Subject" + sub)
    tc = io.loadmat(os.path.join(BASE_DIR, sub, "facesfMRI", "tc_task_parcel500.mat"))
#    tc = tc["tc_roi"]
    tc = tc["tc_parcel"]
    regressors = io.loadmat(os.path.join(BASE_DIR, sub, "facesfMRI", "facesSPM.mat"))
    regressors = regressors['SPM'][0, 0].xX[0, 0].X # Contains task and SHIFTED versions of motion regressors
    # Create feature matrix
    ind = regressors[:, 0] > thresh 
    features = tc[ind, :] # Initialize the shape of features
    labels = np.ones(np.sum(ind))
    indCtrl = np.nonzero(regressors[:, 10] > thresh)
    features = np.vstack((features, tc[indCtrl[0][0:n_samp_trial], :]))
    labels = np.hstack((labels, -np.ones(n_samp_trial)))
    for i in np.arange(4) + 1: # 5 blocks of angry faces, 5 blocks of neutral faces
        ind = regressors[:, i] > thresh
        features = np.vstack((features, tc[ind, :]))
        labels = np.hstack((labels, np.ones(np.sum(ind))))        
        features = np.vstack((features, tc[indCtrl[0][(i * n_samp_trial):(n_samp_trial + i * n_samp_trial)], :]))
        labels = np.hstack((labels, -np.ones(n_samp_trial)))
    n_samp_class = features.shape[0] / 2
    
#    # Normalize the features    ##### Have to normalize the features properly i.e. within cross val
    features -= np.mean(features, axis=0)
#    features /= np.std(features, axis=0)
    
    
    K = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI", "K_rest_anatDense_parcel500_quic335_cv.mat"))            
    K = K['Krest']
    W, V = np.linalg.eig(K)
    K = np.dot(np.dot(V, np.diag(np.sqrt(W))), V.T)    
    K = np.eye(K.shape[0])
    folds = KFold(n_samp_class * 2, n_folds, indices=False)
    scores = []
    for train, test in folds:
        ifolds = KFold(np.sum(train), n_ifolds, indices=False)        
        if method < 4:        
            clf = grid_search.GridSearchCV(classifier, parameters, cv=ifolds)
            clf.fit(features[train, :], labels[train, :])
            print clf.best_estimator_.C
            scores.append(clf.best_estimator_.score(features[test, :], labels[test]))
        else:
            for C in parameters:
                iscores = []                
                for itrain, itest in ifolds:
                    X = np.vstack((features[itrain, :], 1 / np.sqrt(C) * K))
                    Y = np.hstack((labels[itrain], np.zeros(K.shape[0])))
                    classifier.fit(X, Y)
                    iscores.append(classifier.score(features[itest, :], labels[itest]))
                if C == parameters[0]:
                    iscores_acc = iscores
                else:
                    iscores_acc = np.vstack((iscores_acc, iscores))
            C_opt = parameters[np.argmax(np.mean(iscores_acc, axis=1))]
            print C_opt
            X = np.vstack((features[train, :], 1 / np.sqrt(C_opt) * K))            
            Y = np.hstack((labels[train], np.zeros(K.shape[0])))
            classifier.fit(X, Y)
            scores.append(classifier.score(features[test, :], labels[test]))
    print scores
        
        
        
    
    
    

