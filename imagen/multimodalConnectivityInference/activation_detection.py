# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:00:07 2012

@author: bn228083
"""
import os
import numpy as np
from scipy import io
from scipy import linalg
from sklearn.covariance import oas
from bayes_regression import bayesian_regression

BASE_DIR = "/volatile/bernardng/data/imagen/"
TR = 2.2
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectList.txt"), dtype='str')

for sub in [subList[0]]:
    tc_task = io.loadmat(os.path.join(BASE_DIR, sub, "gcafMRI", "tc_task_parcel500.mat"))
    Y = tc_task["tc_parcel"]
    regressors = io.loadmat(os.path.join(BASE_DIR, sub, "gcafMRI", "gcaSPM.mat"))
    regressors = regressors['SPM'][0, 0].xX[0, 0].X # Contains task and SHIFTED versions of motion regressors
    n_cond = 10    
    X = regressors[:, 0:n_cond]
    tc_rest = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI", "tc_rest_parcel500.mat"))
    tc_rest = tc_rest["tc_parcel"]
    S, _ = oas(tc_rest)
    K = linalg.inv(S)
    beta = bayesian_regression(X, Y, K)
    
