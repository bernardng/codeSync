"""
Generate parcel timecourses for IMAGEN data
"""
import os
import numpy as np
from scipy import io
from nipy.labs import as_volume_img

BASE_DIR = "/volatile/bernardng/data/imagen/"
TR = 2.2
#subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectList.txt"), dtype='str')
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/facesList.txt"), dtype='str')
data_type = 0 # 0 = task, # 1 = rest

# Load parcel template
template = io.loadmat(os.path.join(BASE_DIR, "group/parcel500refined.mat"))
template = template['template'].ravel()
label = np.unique(template)
brain_img = as_volume_img("/volatile/bernardng/templates/spm8/rgrey.nii") # For resample the tissue maps
for sub in subList:
    print str("Subject" + sub)
    # Load preprocessed voxel timecourses    
    if data_type:    
        tc = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI/tc_rest_vox.mat"))
    else:
#        tc = io.loadmat(os.path.join(BASE_DIR, sub, "gcafMRI/tc_task_vox.mat"))
        tc = io.loadmat(os.path.join(BASE_DIR, sub, "facesfMRI/tc_task_vox.mat"))
    tc = tc["tc_vox"]
    # Generate tissue mask
    gm_file = os.path.join(BASE_DIR, sub, "anat", "gmMask.nii")
    gm_img = as_volume_img(gm_file)
    gm_img = gm_img.resampled_to_img(brain_img)
    gm = gm_img.get_data()
    wm_file = os.path.join(BASE_DIR, sub, "anat", "wmMask.nii")
    wm_img = as_volume_img(wm_file)
    wm_img = wm_img.resampled_to_img(brain_img)
    wm = wm_img.get_data()
    csf_file = os.path.join(BASE_DIR, sub, "anat", "csfMask.nii")
    csf_img = as_volume_img(csf_file)
    csf_img = csf_img.resampled_to_img(brain_img)
    csf = csf_img.get_data()
    probTotal = gm + wm + csf
    ind = probTotal > 0
    gm[ind] = gm[ind] / probTotal[ind]
    wm[ind] = wm[ind] / probTotal[ind]
    csf[ind] = csf[ind] / probTotal[ind]
    tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
    tissue_mask = tissue.argmax(axis=0) + 1
    tissue_mask[probTotal.ravel() == 0] = 0
    # Generate parcel timecourses   
    tc_parcel = np.zeros((tc.shape[0], label.shape[0] - 1))
    for i in np.arange(label.shape[0] - 1): # Skipping background
        ind = (template == label[i + 1]) & (tissue_mask == 1)
        tc_parcel[:, i] = np.mean(tc[:, ind], axis=1)
    if data_type:
        io.savemat(os.path.join(BASE_DIR, sub, "restfMRI", "tc_rest_parcel500.mat"), {"tc_parcel": tc_parcel})
    else:
#        io.savemat(os.path.join(BASE_DIR, sub, "gcafMRI", "tc_task_parcel500.mat"), {"tc_parcel": tc_parcel})
        io.savemat(os.path.join(BASE_DIR, sub, "facesfMRI", "tc_task_parcel500.mat"), {"tc_parcel": tc_parcel})