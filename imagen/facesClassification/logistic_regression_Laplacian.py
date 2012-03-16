"""
Small scale logistic regression with Laplacian penalty
Input:  X = NxD feature matrix, N = #samples, D = #features
        t = NxK label matrix, N = #samples, K = #classes
        L = DxD Laplacian matrix
        alpha = level of sparse penalty
        n_iter = #iterations
Output: y = NxK predicted label matrix
Note: Optimized using Conjugate Gradient
"""

import numpy as np

def log_regress_Laplacian(X, t, L, alpha=0, n_iter=1000, tol=1e-6):
    n_samp, n_feat = X.shape
    w_old = np.dot(np.linalg.pinv(X), t)
    Xw = np.zeros(n_samp)
    g_old = np.zeros(n_feat)
    u = np.zeros(n_feat)
    for k in np.arange(n_iter):
        g = np.dot(sigmoid(-t * Xw), np.tile(t, (n_feat, 1)).T * X) - alpha * np.hstack((np.dot(w_old[:-1], L), w_old[-1]))#np.zeros(1)))
        if k > 0:        
            beta = np.dot(g, (g - g_old)) / np.dot(u, (g - g_old))
        else:
            beta = 1
        u = g - beta * u
        z = np.dot(g, u) / (alpha * np.dot(u[:-1], np.dot(u[:-1], L)) + np.dot(sigmoid(Xw) * sigmoid(-Xw), np.dot(X, u)**2))
        w = w_old + z * u

#        print np.abs(np.sum(w - w_old)) / np.abs(np.sum(w))
        if np.abs(np.sum(w - w_old)) / np.abs(np.sum(w)) < tol:
            break
        else:
            w_old = w
            Xw = np.dot(X, w)
            g_old = g

    print "Iteration %d" % k
    return w


def sigmoid(a):
    return 1. / (1. + np.exp(-a))
    
def loss(X, w, t, alpha):
    y = sigmoid(np.dot(X, w))
    return -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) + alpha * np.dot(w, w)