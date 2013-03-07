"""
Parcel centroid computation
Input:  template_file = location of parcel template (.nii)
Output: parcel_centroid = centroid of each parcel
"""
import os
import numpy as np
import argparse
from nipy.labs import as_volume_img
from scipy import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute parcel centroid.')
    parser.add_argument('-i', dest='filepath')
    filepath = parser.parse_args().filepath
    path, afile = os.path.split(filepath)
    afile = afile.replace('.nii','_coords.mat')
    template_img = as_volume_img(filepath)
    template = template_img.get_data()
    rois = np.unique(template)
    rois = rois[1:] # Remove background
    n_rois = np.size(rois)
    
    masks = template[np.newaxis, :] == np.arange(n_rois + 1)[:, np.newaxis, np.newaxis, np.newaxis] # Each row = mask of an ROI
    grid = np.mgrid[0:template.shape[0], 0:template.shape[1], 0:template.shape[2]] # Three 3D volumes of x, y, and z coords
    roi_cen = (grid[np.newaxis, :] * masks[:, np.newaxis, :]).reshape(n_rois + 1, 3, -1).sum(axis=-1) / masks.reshape(n_rois + 1, -1).sum(axis=-1).astype(np.float64)[:, np.newaxis]
    roi_cen = roi_cen[1:] # Compute centroid of each ROI
    roi_cen = np.vstack((roi_cen.T, np.ones(n_rois))
    parcel_cen_mni = np.dot(template_img.affine, roi_cen)
    parcel_cen_mni = parcel_cen_mni[0:3].T
    io.savemat(os.path.join(path, afile), {"coords": parcel_cen_mni})


