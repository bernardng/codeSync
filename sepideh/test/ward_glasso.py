############################################################################
# Ward Clustering and Graphical LASSO on preprocessed sepideh's data
############################################################################
import numpy as np
import pylab as pl
from scipy import linalg, ndimage
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import WardAgglomeration
from sklearn.covariance import GraphLassoCV, OAS
from sklearn.preprocessing import Scaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.externals.joblib import Memory
from sklearn.cross_validation import KFold

tc = time_series_gm
cv = KFold(tc.shape[0], 3)
glasso = GraphLassoCV(verbose=1, n_refinements=3, alphas=3)
mem = Memory(cachedir='.', verbose=1)
A = grid_to_graph(n_x=gm_mask.shape[0], n_y=gm_mask.shape[1], n_z=gm_mask.shape[2], mask=gm_mask)
ward = WardAgglomeration(n_clusters=100, connectivity=A, memory=mem, n_components=1)
scaler = Scaler()
pipe = Pipeline([('ward', ward), ('scaler', scaler), ('glasso', glasso)])
clf = GridSearchCV(pipe, {'ward__n_clusters': [100, 200, 500]}, n_jobs=1, verbose=1)
clf.fit(tc)
fitted_glasso = clf.best_estimator.named_steps['glasso']
cov_ = fitted_glasso.covariance_
prec_ = fitted_glasso.precision_
