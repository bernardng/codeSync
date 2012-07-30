"""
Bayesian Regression
Input:  X = nxm regressor matrix, n = #samples, m = #regressors
        Y = nxd data matrix, d = #features
        K = dxd prior precision
Output: beta = dxm posterior regression coefficients
"""
import numpy as np
from .model_evidence import model_evidence
from scipy import linalg

def bayesian_regression(X, Y, K):
    d = K.shape[0]
    alpha = model_evidence(X, Y, K)
    V1_inv = np.eye(d)
    V2_inv = K
    # beta = (V1inv+alpha*V2inv)\(V1inv*Y'*X)/(X'*X)
    YTX = np.dot(Y.T, X)    
    XTX = np.dot(X.T, X)
    beta = np.dot(np.dot(linalg.pinv(V1_inv + alpha * V2_inv), 
                         np.dot(V1_inv, YTX)), linalg.pinv(XTX))
    return beta


