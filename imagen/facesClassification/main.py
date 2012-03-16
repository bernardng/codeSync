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
from logistic_regression_Laplacian import log_regress_Laplacian, sigmoid

BASE_DIR = "/volatile/bernardng/data/imagen/"
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/facesList.txt"), dtype='str')

thresh = 0.56
n_samp_trial = 8 # Manually determined
n_folds = 5
n_ifolds = 4

method = 4

if method == 0:
    # Logistic Regression
    classifier = LogisticRegression(penalty='l2', scale_C=True)
    parameters = {'C':[1e50]}
elif method == 1:
    # SVM
    classifier = svm.LinearSVC(scale_C=True)
#    parameters = {'C':np.logspace(0, 6, 20)}
    parameters = {'C':np.linspace(1e0, 1e6, 10)}
elif method == 2:
    # l2 regularized Logistic Regression
    classifier = LogisticRegression(penalty='l2', scale_C=True)
#    parameters = {'C':np.logspace(0, 6, 20)}
    parameters = {'C':np.linspace(1e0, 1e6, 10)}
#    parameters = {'C':[1e4]}
elif method == 3:
    # Sparse Logistic Regression
    classifier = LogisticRegression(penalty='l1', scale_C=True)
#    parameters = {'C':np.logspace(0, 6, 30)}
    parameters = {'C':np.linspace(1e0, 1e6, 10)}
elif method == 4:
    # Connectivity-informed Logistic Regression
    parameters = np.linspace(1e0, 1e6, 10)
#    parameters = [1e50, 1e49]

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
    labels = np.ones(n_samp_trial)
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
    
    K = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI", "K_rest_anatDense_parcel500_quic335_cv.mat"))            
    K = K['Krest']
    L = K
    L = np.eye(K.shape[0])
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
#                    alpha = np.sum(train[itrain]) / C # scaling alpha by the number of samples                
#                    w = log_regress_Laplacian(np.hstack((features[train[itrain]], np.ones((np.sum(train[itrain]), 1)))), labels[train[itrain]], L, alpha=alpha, n_iter=500, tol=1e-6)
#                    y = np.sign(np.dot(np.hstack((features[train[itest]], np.ones((np.sum(train[itest]), 1)))), w))
#                    iscores.append(np.mean(y == labels[train[itest]]))
                    classifier = LogisticRegression(penalty='l2', C=C / np.sum(train[itrain]), scale_C=True)
                    classifier.fit(features[train[itrain]], labels[train[itrain]])
                    iscores.append(classifier.score(features[train[itest]], labels[train[itest]]))
                if C == parameters[0]:
                    iscores_acc = iscores
                else:
                    iscores_acc = np.vstack((iscores_acc, iscores))
            C_opt = parameters[np.argmax(np.mean(iscores_acc, axis=1))]
            print C_opt
#            alpha = np.sum(train) / C_opt
#            w = log_regress_Laplacian(np.hstack((features[train], np.ones((np.sum(train), 1)))), labels[train], L, alpha=alpha, n_iter=500, tol=1e-6)
#            y = np.sign(np.dot(np.hstack((features[test], np.ones((np.sum(test), 1)))), w))
#            scores.append(np.mean(y == labels[test]))
            classifer = LogisticRegression(penalty='l2', C=C_opt / np.sum(train), scale_C=True)
            classifier.fit(features[train], labels[train])
            scores.append(classifier.score(features[test], labels[test]))
    print scores
        
        
        
    
    
    

