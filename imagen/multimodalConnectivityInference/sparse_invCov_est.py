############################################################################
# Graphical LASSO on Freesurfer clusters 
############################################################################
import os
import numpy as np
import pylab as pl
from scipy import io
from parietal.time_series import preprocessing
from sklearn.covariance import GraphLassoCV, OAS

BASE_DIR = "/volatile/bernardng/data/imagen/"
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectList.txt"), dtype='str')

for sub in subList:
tc_roi = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI", "tc_rest_roi_fs.mat"))
#    tc_roi = io.loadmat(os.path.join(BASE_DIR, sub, "gcafMRI/tc_task_roi_fs.mat"))
tc_roi = tc_roi['tc_roi']
tc_roi = np.array(tc_roi[1:,:]) # Skipping first timepoint to create 3 even folds

# Normalization
tc_roi -= np.mean(tc_roi, axis=0)
tc_roi /= np.std(tc_roi, axis=0)

#    tc_roi = preprocessing.standardize(tc_roi.T).T
#    ind_nan = np.isnan(tc_roi)    
#    if np.sum(ind_nan) > 0:
#        tc_roi[ind_nan] = np.random.randn(ind_nan.shape[0]) # Putting in random signals for zero timecourses

glasso = GraphLassoCV(verbose=1, n_refinements=5, alphas=5, tol=1e-6, n_jobs=1)
glasso.fit(tc_roi)
alpha_ = glasso.alpha_    
cov_ = glasso.covariance_
K_rest = glasso.precision_

#    io.savemat(os.path.join(BASE_DIR, sub, "precMat", "K_rest_roi_fs_GV55.mat"), {"K_rest": K_rest})
    
    