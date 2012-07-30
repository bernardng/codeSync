"""
Estimate optimal level of regularization for Bayesian Regression Model
Input:  X = nxm regressor matrix, n = #samples, m = #regressors
        Y = nxd data matrix, d = #features
        K = dxd prior precision
Output: alpha = optimal amount of regularization
        evid = model evidence
"""
import numpy as np
from scipy import linalg
from scipy import optimize

def model_evidence(X, Y, K):
    n, m = X.shape
    d = K.shape[0]
    eigval, eigvec = linalg.eig(K)
    # B = eigvec'*Y'*X/(X'*X)*X'*Y*eigvec    
    ETYTX = np.dot(np.dot(eigvec.T, Y.T), X)
    B = np.dot(np.dot(ETYTX, linalg.pinv(np.dot(X.T, X))), ETYTX.T)
    alpha = optimize.fmin_bfgs(f, 0.01, args=(m, d, eigval, B))
    return alpha

def f(alpha, m, d, eigval, B):
    return np.double(-m * d / 2 * np.log(alpha) + m / 2 * np.sum(np.log(1 
        + alpha * eigval)) - 0.5 * np.sum(np.diag(B) / (1 + alpha * eigval)))
    
    
    


