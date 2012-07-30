"""
Max-t permutation test
Input:  beta = mxdxs regression coefficients, m = #conditions, d = #features, s = #samples  
        contrast_list = cx2 list of contrasts, c = #contrast, e.g. {[1,2,3],[4,5]}
        thresh = tx1 threshold percentile, e.g. 0:0.25:100
        n_perm = number of permutation
Output: sig = cxdxt binary map of significant
Ref:    Nichols and Hayasaka, Controlling the familywise error rate in
        functional neuroimaging: a comparative review, Stat. Methods Med. Research, 2003
"""
import numpy as np
from scipy import stats

def max_t_perm_test(beta, contrast_list, thresh, n_perm):
    m, d, s = np.shape(beta)    
    c = np.shape(contrast_list)[0]
    t = np.shape(thresh)[0]
    tval = np.zeros((c, d))
    sig = np.zeros((c, d, t))
    for i in np.arange(c):
        cond1 = contrast_list[i][0].ravel()
        if cond1.any():
            beta1 = np.squeeze(np.mean(beta[cond1 - 1, :, :], axis=0)).T # sxd            
        else:
            beta1 = np.zeros((s, d))            
        cond2 = contrast_list[i][1].ravel()
        if cond2.any():
            beta2 = np.squeeze(np.mean(beta[cond2 - 1, :, :], axis=0)).T # sxd
        else:
            beta2 = np.zeros((s, d))            
        tval[i, :] = stats.ttest_rel(beta1, beta2)[0]

        tval_perm = np.zeros((n_perm, d))
        for perm in np.arange(n_perm):
            sgn = np.sign(np.random.randn(s))
            tval_perm[perm, :] = stats.ttest_rel(sgn[:, np.newaxis] * beta1, sgn[:, np.newaxis] * beta2)[0]
        max_tval = np.max(tval_perm, axis=1)
        
        for j in np.arange(t):
            sig[i, :, j] = tval[i, :] > stats.scoreatpercentile(max_tval, thresh[j])
        print 'Contrast' + np.str(i)
        
    return sig
        
        
        
        
        
    
