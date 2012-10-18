"""
Functional Parcellation for IMAGEN data
Input:	n_parcels = #parcels to divide the brain
	BASE_DIR = location where data are stored
	subList = subject#'s
	ANAT_DIR = location of anatomical template
Notes:	Requires folder structure described in readMe.txt
"""
import os
import numpy as np
import nibabel as nib
from scipy import io
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from nipy.labs import as_volume_img
from sklearn.decomposition import PCA
from sklearn.externals.joblib import Memory
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import WardAgglomeration

# Choose number of parcels
n_parcels = 500.0

# Change path to files
BASE_DIR = "/media/GoFlex/research/data/imagen/"
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectList.txt"), dtype='str')
ANAT_DIR = "/media/GoFlex/research/templates/freesurfer/cort_subcort_cerebellum_333.nii"

# Concatenating PCA-ed voxel timecourses across subjects
for sub in subList:
    tc = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI/tc_vox.mat"))
    tc = tc["tc"]
    pca = PCA(n_components=10)    
    pca.fit(tc.T)
    tc_pca = pca.transform(tc.T)
    
    # Standardizing pca-ed time courses
    tc_std = np.std(tc_pca, axis=1)
    ind = tc_std > 1e-16
    tc_pca[ind, :] = tc_pca[ind, :] - tc_pca[ind, :].mean(axis=1)[:, np.newaxis]
    tc_pca[ind, :] = tc_pca[ind, :] / tc_pca[ind, :].std(axis=1)[:, np.newaxis]

    # Concatenate time courses across subjects
    if sub == subList[0]:
        tc_group = tc_pca
    else:
        tc_group = np.hstack((tc_group, tc_pca))
    print("Concatenating subject" + sub + "'s timecourses")

# Load Freesurfer template
brain_img = as_volume_img(ANAT_DIR)

# Run relabel_disconnected_rois.py

brain = brain_img.get_data()
dim = np.shape(brain)
rois = np.unique(brain)
rois = rois[1:] # Remove background
n_rois = np.shape(rois)[0]
n_vox = np.sum(brain != 0) # Number of voxels within ROIs in Freesurfer template

# Spatially smoothing the group timecourses
tc_group = tc_group.reshape((dim[0], dim[1], dim[2], -1))
n_tpts = tc_group.shape[-1]
for t in np.arange(n_tpts):
    tc_group[:,:,:,t] = gaussian_filter(tc_group[:,:,:,t], sigma=1.5)
tc_group = tc_group.reshape((-1, n_tpts))

# Perform parcellation on smoothed PCA-ed timecourses for each ROI
mem = Memory(cachedir='.', verbose=1)
n_clust = np.zeros(n_rois) # Different #clusters for different ROI
template = np.zeros((dim[0], dim[1], dim[2]))
print("Performing Ward Clustering")
for i in np.arange(n_rois):
    # Determine the number of clusters to divide each ROI into
    roi_mask = brain == rois[i]
    n_clust[i] = np.round(np.sum(roi_mask) * n_parcels / n_vox)
    if n_clust[i] <= 1:
        template[roi_mask] = np.shape(np.unique(template))[0]
    else:        
        # Define connectivity based on brain mask
        A = grid_to_graph(n_x=dim[0], n_y=dim[1], n_z=dim[2], mask=roi_mask)
        # Create ward object
        ward = WardAgglomeration(n_clusters=n_clust[i], connectivity=A.tolil(), memory=mem)
        ward.fit(tc_group[roi_mask.ravel(), :].T)
        template[roi_mask] = ward.labels_ + np.shape(np.unique(template))[0] 

# Run relabel_disconnected_parcel.py

# Saving the template
io.savemat(os.path.join(BASE_DIR, "group/fs_w_cer_parcel500.mat"), {"template":template})
nii = nib.Nifti1Image(template, brain_img.affine)
nib.save(nii, os.path.join(BASE_DIR, "group/fs_w_cer_parcel500.nii"))         

# Remove parcels with zero timecourses in any of the subjects
template = template.ravel()
template_refined = template.copy()
label = np.unique(template)
for sub in subList:
    print str("Subject" + sub)
    # Load preprocessed voxel timecourses    
    tc = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI/tc_vox.mat"))
    tc = tc["tc"]
   
    # Generate subject-specific tissue mask
    gm_file = os.path.join(BASE_DIR, sub, "anat", "gmMask.nii")
    gm_img = as_volume_img(gm_file)
    gm_img = gm_img.resampled_to_img(brain_img)
    gm = gm_img.get_data()
    gm[gm < 0] = 0
    wm_file = os.path.join(BASE_DIR, sub, "anat", "wmMask.nii")
    wm_img = as_volume_img(wm_file)
    wm_img = wm_img.resampled_to_img(brain_img)
    wm = wm_img.get_data()
    wm[wm < 0] = 0
    csf_file = os.path.join(BASE_DIR, sub, "anat", "csfMask.nii")
    csf_img = as_volume_img(csf_file)
    csf_img = csf_img.resampled_to_img(brain_img)
    csf = csf_img.get_data()
    csf[csf < 0] = 0
    probTotal = gm + wm + csf
    ind = probTotal > 0
    gm[ind] = gm[ind] / probTotal[ind]
    wm[ind] = wm[ind] / probTotal[ind]
    csf[ind] = csf[ind] / probTotal[ind]
    tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
    tissue_mask = tissue.argmax(axis=0) + 1
    tissue_mask[probTotal.ravel() == 0] = 0
    
    # Refine Parcellation    
    tc_parcel = np.zeros((tc.shape[0], label.shape[0] - 1))
    for i in np.arange(label.shape[0] - 1): # Skipping background
        ind = (template == label[i + 1]) & (tissue_mask == 1)
        tc_parcel[:, i] = np.mean(tc[:, ind], axis=1)
        if np.sum(tc_parcel[:, i]) == 0:
            template_refined[template == label[i + 1]] = 0
template_refined = template_refined.reshape([dim[0], dim[1], dim[2]])

# Ensure template labels do not have gaps in the numbers, e.g. 0 1 3 ...
rois = np.unique(template_refined)
template_refined = (rois[:, np.newaxis, np.newaxis, np.newaxis] == template_refined[np.newaxis, :]).astype(int).argmax(0)

# Saving template_refined
io.savemat(os.path.join(BASE_DIR, "group/fs_w_cer_parcel500_refined.mat"), {"template": template_refined})
nii = nib.Nifti1Image(template_refined, brain_img.affine)
nib.save(nii, os.path.join(BASE_DIR, "group/fs_w_cer_parcel500_refined.nii"))            

