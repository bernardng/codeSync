"""
Activation detection
"""
import os
import numpy as np
from scipy import io
from scipy import linalg
from sklearn.covariance import oas
from bayesian_regression_BN import bayesian_regression
from bayesian_regression_BN import max_t_perm_test

BASE_DIR = "/volatile/bernardng/data/imagen/"
TR = 2.2
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectListDWI.txt"), dtype='str')
n_subs = subList.shape[0]
n_conds = 10

method = 1

if method == 1:
    methName = 'OLS'
elif method == 2:
    methName = 'Ridge'
elif method == 3:
    methName = 'OAS'
elif method == 4:
    methName = 'SGGM'
elif method == 5:
    methName = 'WSGGM'
elif method == 6:
    methName = 'DWI'
elif method == 7:
    methName = 'sketch'
thresh = np.arange(0, 100, 0.25)

# Load ROI template
template = io.loadmat(os.path.join(BASE_DIR, "group/fs_parcel500.mat"))
template = template['template']
rois = np.unique(template)
n_rois = np.size(rois) - 1 # Skipping background

beta = np.zeros((n_conds, n_rois, n_subs))
for sub in np.arange(n_subs):
    # Load task data
    tc_task = io.loadmat(os.path.join(BASE_DIR, subList[sub], "gcafMRI", "tc_fs_parcel500.mat"))
    Y = tc_task["tc"]
    # Normalize task data
    Y = Y - np.mean(Y, axis=0)
    Y = Y / np.std(Y, axis=0)
    # Load regressors for task data
    regressors = io.loadmat(os.path.join(BASE_DIR, subList[sub], "gcafMRI", "gcaSPM.mat"))
    regressors = regressors['SPM'][0, 0].xX[0, 0].X # Contains task and SHIFTED versions of motion regressors
    X = regressors[:, 0:n_conds]
    # Normalize regressors
    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)
    # beta estimation (n_conds x n_rois x n_subs)
    if method == 1: # OLS
        beta[:, :, sub] = np.dot(linalg.pinv(X), Y)
    print 'Subject' + subList[sub] + ' beta computed'

# Max-t permutation test for activation inference
contrast_list = io.loadmat(os.path.join(BASE_DIR, "group/contrastList.mat"))
contrast_list = contrast_list['contrastList']
n_perm = 10000
sig = max_t_perm_test(beta, contrast_list, thresh, n_perm)
        
    
#    tc_rest = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI", "tc_rest_parcel500.mat"))
#    tc_rest = tc_rest["tc_parcel"]
#    S, _ = oas(tc_rest)
#    K = linalg.inv(S)
#    beta = bayesian_regression(X, Y, K)
    
