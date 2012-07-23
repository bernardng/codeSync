"""
Parcel graph construction
Input:  template_file = location of parcel template (.nii)
Output: graph_adjacent = binary dxd matrix with 1 for spatially-connected parcels (.mat)
        graph_bilateral = binary dxd matrix with 1 for bilateral parcel pairs (.mat)
        graph_right_ipsi = binary dxd matrix with 1 for spatially-connected parcels on right side of brain (.mat)
        graph_left_ipsi = binary dxd matrix with 1 for spatially-connected parcels on left side of brain (.mat)
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
    n_neigh = 1 # Increase n_neigh to increase robustness to parcellation errors
    
    # Build bilateral connection matrix
    masks = template[np.newaxis, :] == np.arange(n_rois + 1)[:, np.newaxis, np.newaxis, np.newaxis] # Each row = mask of an ROI
    grid = np.mgrid[0:template.shape[0], 0:template.shape[1], 0:template.shape[2]] # Three 3D volumes of x, y, and z coords
    roi_cen = (grid[np.newaxis, :] * masks[:, np.newaxis, :]).reshape(n_rois + 1, 3, -1).sum(axis=-1) / masks.reshape(n_rois + 1, -1).sum(axis=-1).astype(np.float64)[:, np.newaxis]
    roi_cen = roi_cen[1:] # Compute centroid of each ROI
    x_cen = np.mean(grid[0][template > 0]) # x centroid computed at voxel level
    roi_cen_xflip = roi_cen.copy()
    roi_cen_xflip[:, 0] = 2 * x_cen - roi_cen[:, 0] # Find bilateral ROI
    ind_neigh = np.sum((roi_cen[np.newaxis, :] - roi_cen_xflip[:, np.newaxis, :]) ** 2, axis=2).argsort(axis=1)
    graph_bilateral = np.zeros((n_rois, n_rois, n_neigh))
    for n in np.arange(n_neigh):
        graph_bilateral[np.arange(n_rois), ind_neigh[:, n], n] = 1
        graph_bilateral[:, :, n] = (graph_bilateral[:, :, n] + graph_bilateral[:, :, n].T) # Symmeterization
        if n > 0:        
            graph_bilateral[:, :, n] += np.sum(graph_bilateral[:, :, 0:n], axis=2)
        graph_bilateral = np.float64(graph_bilateral > 0)
    # Save bilateral connection matrix    
    io.savemat(os.path.join(path, 'graph_bilateral_' + afile), {"B": graph_bilateral})
    
    # Build adjacency matrix
    graph_adjacent = np.zeros((n_rois, n_rois))
    for nr in np.arange(n_rois):
        # Generate dilated mask for each ROI, in which the border would intersect with adjacent ROIs
        mask = binary_dilation(template == rois[nr], structure=np.ones((2, 2, 2)))
        # Put 1 in adjacency matrix for ROIs that the diated mask intersect
        graph_adjacent[nr, list(set(np.unique(template[mask]) - 1) - set((rois[nr] - 1, -1)))] = 1 # -1 to account for background removal
    graph_adjacent = (graph_adjacent + graph_adjacent.T) > 0 # Symmeterization
    graph_adjacent = np.float64(graph_adjacent * (1 - graph_bilateral[:, :, -1]))
    # Save adjacency matrix    
    io.savemat(os.path.join(path, 'graph_adjacent_' + afile), {"A": graph_adjacent})
    
    # Build ipsilateral connection matrix
    ind_left = np.nonzero(roi_cen[:, 0] > x_cen) # Parcels on left side of brain    
    ind_right = np.nonzero(roi_cen[:, 0] <= x_cen) # Parcels on right side of brain
    # Build left ipsilateral connection matrix
    graph_left_ipsi = np.zeros((n_rois, n_rois))
    for nr in ind_left[0]:
        # Generate dilated mask for each ROI, in which the border would intersect with adjacent ROIs
        mask = binary_dilation(template == rois[nr], structure=np.ones((2, 2, 2)))
        # Put 1 in adjacency matrix for ROIs that the diated mask intersect
        graph_left_ipsi[nr, list(set(np.unique(template[mask]) - 1) - set((rois[nr] - 1, rois[ind_right[0]] - 1, -1)))] = 1 # -1 to account for background removal
    graph_left_ipsi = np.float64((graph_left_ipsi + graph_left_ipsi.T) > 0) # Symmeterization
    # Save left ipsilater connection matrix    
    io.savemat(os.path.join(path, 'graph_left_ipsi_' + afile), {"L": graph_left_ipsi})
    
    # Build right ipsilateral connection matrix
    graph_right_ipsi = np.zeros((n_rois, n_rois))
    for nr in ind_right[0]:
        # Generate dilated mask for each ROI, in which the border would intersect with adjacent ROIs
        mask = binary_dilation(template == rois[nr], structure=np.ones((2, 2, 2)))
        # Put 1 in adjacency matrix for ROIs that the diated mask intersect
        graph_right_ipsi[nr, list(set(np.unique(template[mask]) - 1) - set((rois[nr] - 1, rois[ind_left[0]] - 1, -1)))] = 1 # -1 to account for background removal
    graph_right_ipsi = np.float64((graph_right_ipsi + graph_right_ipsi.T) > 0) # Symmeterization
    # Save right ipsilater connection matrix    
    io.savemat(os.path.join(path, 'graph_right_ipsi_' + afile), {"R": graph_right_ipsi})


    
    
