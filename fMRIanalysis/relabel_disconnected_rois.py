# Relabel ROIs with spatially disconnected components
# Input:    brain = label template with 1 = graymatter voxels, 2 onwards = ROIs
# Output:   brain_relabel = relabeled template
import numpy as np
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_closing, binary_dilation, generate_binary_structure

#def relabel_disconnected_rois(brain):
brain_relabel = brain.copy()
while True:
    split_roi = 0
    roi = np.unique(brain_relabel)
    for i in roi[2:]:
	labels, n_labels = label(brain_relabel == i, structure=generate_binary_structure(3, 1))
        if n_labels > 1:
#	    max_n_vox = 0 # Label of the largest component stays fixed
#            for j in np.unique(labels)[1:]:
#		if np.sum(labels == j) > max_n_vox:
#		    max_n_vox = np.sum(labels == j)
#		    max_label = j
#	    if np.sum(labels == max_label) > 5:	    
#		labels[labels == max_label] = 0
	    for j in np.unique(labels)[1:]:
		if np.sum(labels == j) < 6:
	            brain_relabel[labels == j] = 1 # Assign to GM set
		else:
#		    brain_relabel[(labels >= j) | (brain_relabel > i)] += 1
		    brain_relabel[(labels > j) | (brain_relabel > i)] += 1
		    split_roi = 1
		    break
	if split_roi == 1:
	    break
    if split_roi == 0:
	break
#    return brain_relabel

