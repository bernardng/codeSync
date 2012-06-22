"""
Parcel graph construction
Input:  template_file = location of parcel template (.nii)
Output: graph_adjacent = binary dxd matrix with 1 for spatially-connected parcels (.mat)
        graph_bilateral = binary dxd matrix with 1 for bilateral parcel pairs (.mat)
"""
import os
import numpy as np
import argparse
from nipy.labs import as_volume_img
from scipy import io
from scipy.ndimage.morphology import binary_dilation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adjacency and bilateral graph.')
    parser.add_argument('-i', dest='filepath')
    filepath = parser.parse_args().filepath
    path, afile = os.path.split(filepath)
    afile = afile.replace('.nii','.mat')
    template_img = as_volume_img(filepath)
    template = template_img.get_data()
    rois = np.unique(template)
    rois = rois[1:] # Remove background
    n_rois = np.size(rois)
    
    # Build adjacency matrix
    graph_adjacent = np.zeros((n_rois, n_rois))
    for nr in np.arange(n_rois):
        # Generate dilated mask for each ROI, in which the border would intersect with adjacent ROIs
        mask = binary_dilation(template == rois[nr], structure=np.ones((2, 2, 2)))
        # Put 1 in adjacency matrix for ROIs that the diated mask intersect
        graph_adjacent[nr, list(set(np.unique(template[mask]) - 1) - set((rois[nr] - 1, -1)))] = 1 # -1 to account for background removal
    graph_adjacent = (graph_adjacent + graph_adjacent.T) > 0 # Symmeterization

    # Save adjacency matrix    
    io.savemat(os.path.join(path, 'graph_adjacent_' + afile), {"A": graph_adjacent})
    
    # Build bilateral connection matrix
    masks = template[np.newaxis, :] == np.arange(n_rois + 1)[:, np.newaxis, np.newaxis, np.newaxis] # Each row = mask of an ROI
    grid = np.mgrid[0:template.shape[0], 0:template.shape[1], 0:template.shape[2]] # Three 3D volumes of x, y, and z coords
    roi_cen = (grid[np.newaxis, :] * masks[:, np.newaxis, :]).reshape(n_rois + 1, 3, -1).sum(axis=-1) / masks.reshape(n_rois + 1, -1).sum(axis=-1).astype(np.float64)[:, np.newaxis]
    roi_cen = roi_cen[1:] # Compute centroid of each ROI
    roi_cen_xflip = roi_cen.copy()
    roi_cen_xflip[:, 0] = 2 * np.mean(roi_cen[:, 0]) - roi_cen[:, 0] # Find bilateral ROI
    ind_neigh = np.sum((roi_cen[np.newaxis, :] - roi_cen_xflip[:, np.newaxis, :]) ** 2, axis=2).argmin(axis=1)
    graph_bilateral = np.zeros((n_rois, n_rois))
    graph_bilateral[np.arange(n_rois), ind_neigh] = 1
    graph_bilateral = (graph_bilateral + graph_bilateral.T) > 0 # Symmeterization
    
    # Save bilateral connection matrix    
    io.savemat(os.path.join(path, 'graph_bilateral_' + afile), {"B": graph_bilateral})