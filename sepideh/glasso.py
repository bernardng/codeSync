############################################################################
# Graphical LASSO on Yeo's clusters extracted from sepideh's data 
############################################################################
import numpy as np
import pylab as pl
from sklearn.covariance import GraphLassoCV, OAS
tc = tc_roi
glasso = GraphLassoCV(verbose=1, n_refinements=3, alphas=3, n_jobs=2)
glasso.fit(tc)
cov_ = glasso.covariance_
prec_ = glasso.precision_
