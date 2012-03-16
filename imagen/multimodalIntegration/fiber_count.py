# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:39:40 2012

@author: bn228083
"""
from scipy import io
import numpy as np
from enthought.mayavi import mlab

fiber = io.loadmat('/volatile/bernardng/data/imagen/000000112288/dwi/fibers.mat')

n_fibers = fiber['fiber'][0, 0].__dict__['fiber'].shape[1]
XYZ = np.array([0, 0, 0])
XYZ = XYZ[None, :]
for i in np.arange(0, n_fibers, 50):
    xyz = fiber['fiber'][0, 0].__dict__['fiber'][0,0].__dict__['xyzFiberCoord']
    XYZ = np.append(XYZ, np.array(xyz), axis=0)
mlab.points3d(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    