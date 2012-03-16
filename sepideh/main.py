"""
Main function for analyzing Sepideh's data
"""
import os
import numpy as np
from nipy.labs import as_volume_img
from sklearn.covariance import GraphLassoCV, OAS

# Choose subject to test
s = 1

# Constants
BASE_DIR = "/volatile/bernardng/data/sepideh/preprocessedBN/"
TR = 2.4
subList = np.loadtxt(os.path.join(BASE_DIR, "subListTruncated"), dtype='str')
for sub in [subList[s]]:
    # Define path to data
    tc_file = os.path.join(BASE_DIR, sub, "rsfMRI", "wbold_audiospont_1.hdr")
    gm_file = os.path.join(BASE_DIR, sub, "anat", "wc1anat_" + sub + "_3T_neurospin.hdr")
    wm_file = os.path.join(BASE_DIR, sub, "anat", "wc2anat_" + sub + "_3T_neurospin.hdr")
    csf_file = os.path.join(BASE_DIR, sub, "anat", "wc3anat_" + sub + "_3T_neurospin.hdr")
    motion_confounds = np.loadtxt(os.path.join(BASE_DIR, sub, "rsfMRI", "rp_bold_audiospont_1_0001.txt"))
    template_file = "/volatile/bernardng/templates/yeo_sulci_merged/Yeo_cort_sulci_subcort_MNI152_3mm.nii"
    # Load data objects
    tc_img = as_volume_img(tc_file)
    gm_img = as_volume_img(gm_file)
    wm_img = as_volume_img(wm_file)
    csf_img = as_volume_img(csf_file)
    template_img = as_volume_img(template_file)
    # Temporal filtering and removing WM and CSF signal   
    tc, tc_roi = preproc(tc_img, gm_img, wm_img, csf_img, motion_confounds, template_img=template_img, tr=TR)
    
    # Graphical LASSO
    
#    glasso = GraphLassoCV(verbose=1, n_refinements=5, alphas=5, n_jobs=1)
#    glasso.fit(tc_roi)
#    cov_ = glasso.covariance_
#    prec_ = glasso.precision_
    


