# Relabel parcels with spatially disconnected components
# Input:    brain = label template 
# Output:   brain_relabel = relabeled template
import numpy as np
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_closing, binary_dilation, generate_binary_structure

template_relabel = template.copy()
while True:
    split_roi = 0
    parcel = np.unique(template_relabel)
    for i in parcel[1:]:
	labels, n_labels = label(template_relabel == i, structure=generate_binary_structure(3, 1))
        if n_labels > 1:
	    max_n_vox = 0 # Label of the largest component stays fixed
            for j in np.unique(labels)[1:]:
		if np.sum(labels == j) > max_n_vox:
		    max_n_vox = np.sum(labels == j)
		    max_label = j
	    labels[labels == max_label] = 0
	    for j in np.unique(labels)[1:]:
		if np.sum(labels == j) < 10:
	            template_relabel[labels == j] = 0
		else:
		    template_relabel[(labels >= j) | (template_relabel > i)] += 1
		    split_roi = 1
		    break
	if split_roi == 1:
	    break
    if split_roi == 0:
	break

