"""
Functional Parcellation for IMAGEN data
Input:	n_parcels = #parcels to divide the brain
	BASE_DIR = location where data are stored
	subList = subject#'s
Notes: Requires folder structure described in readMe.txt
"""
import os
import numpy as np
import nibabel as nib
from scipy import io
from scipy.ndimage import gaussian_filter
from nipy.labs import as_volume_img
from nipy.labs.datasets.volumes.volume_img import VolumeImg
#from nipy.labs import mask as mask_utils
from sklearn.decomposition import PCA
from sklearn.externals.joblib import Memory
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import WardAgglomeration
from scipy.ndimage.morphology import binary_closing, binary_dilation, generate_binary_structure
from scipy.ndimage import label

# Choose number of parcels
n_parcels = 150.0

# Change path to files
BASE_DIR = "/media/GoFlex/research/data/imagen/"
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectListDWI.txt"), dtype='str')
REF_DIR = "/media/GoFlex/research/templates/spm8/rgrey.nii" 
GM_DIR = "/media/GoFlex/research/templates/icbm/gm.nii"
WM_DIR = "/media/GoFlex/research/templates/icbm/wm.nii"
CSF_DIR = "/media/GoFlex/research/templates/icbm/csf.nii"

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

# Generate dilated GM mask
ref_img = as_volume_img(REF_DIR) # For resampling to 3mm
ref = ref_img.get_data()
gm_img = as_volume_img(GM_DIR)
gm = gm_img.get_data()
wm_img = as_volume_img(WM_DIR)
wm = wm_img.get_data()
csf_img = as_volume_img(CSF_DIR)
csf = csf_img.get_data()
probTotal = gm + wm + csf
dim = np.shape(probTotal)
ind = probTotal > 0
gm[ind] = gm[ind] / probTotal[ind]
wm[ind] = wm[ind] / probTotal[ind]
csf[ind] = csf[ind] / probTotal[ind]
tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
tissue_mask = tissue.argmax(axis=0) + 1
tissue_mask[probTotal.ravel() == 0] = 0
brain = np.reshape(tissue_mask == 1, (dim[0], dim[1], dim[2]))

brain = gm > 0.15

brain = binary_dilation(brain, structure=generate_binary_structure(3, 1))
brain_img = VolumeImg(brain, affine=gm_img.affine, world_space=None, interpolation='nearest')
brain_img = brain_img.resampled_to_img(ref_img, interpolation='nearest')
brain = brain_img.get_data()
brain = binary_closing(brain, structure=generate_binary_structure(3, 1))
labels, n_labels = label(brain) # Remove isolated voxels
for i in np.unique(labels)[2:]:
   brain[labels == i] = 0

brain_img = as_volume_img(GM_DIR)
brain = brain_img.get_data()

# Spatial smoothing to encourage smooth parcels
dim = np.shape(brain)
tc_group = tc_group.reshape((dim[0], dim[1], dim[2], -1))
n_tpts = tc_group.shape[-1]
for t in np.arange(n_tpts):
    tc_group[:,:,:,t] = gaussian_filter(tc_group[:,:,:,t], sigma=1.5)
tc_group = tc_group.reshape((-1, n_tpts))
tc_group = tc_group[brain.ravel()==1, :]

# Functional parcellation with Ward clustering
print("Performing Ward Clustering")
mem = Memory(cachedir='.', verbose=1)
# Define connectivity based on brain mask
A = grid_to_graph(n_x=brain.shape[0], n_y=brain.shape[1], n_z=brain.shape[2], mask=brain)
# Create ward object
ward = WardAgglomeration(n_clusters=n_parcels, connectivity=A.tolil(), memory=mem)
ward.fit(tc_group.T)
template = np.zeros((dim[0], dim[1], dim[2]))
template[brain==1] = ward.labels_ + 1 # labels start from 0, which is used for background

# Remove single voxels not connected to parcel
for i in np.unique(template)[1:]:
    labels, n_labels = label(template == i, structure=np.ones((3,3,3)))
    if n_labels > 1:
	for j in np.arange(n_labels):
	    if np.sum(labels == j + 1) < 10:
		template[labels == j + 1] = 0

# Saving the template
io.savemat(os.path.join(BASE_DIR, "group/gm_bin_parcel150.mat"), {"template": template})
nii = nib.Nifti1Image(template, brain_img.affine)
nib.save(nii, os.path.join(BASE_DIR, "group/gm_bin_parcel150.nii"))

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
        if np.sum(tc_parcel[:, i]) == 0 or np.isnan(np.sum(tc_parcel[:, i])):
            template_refined[template == label[i + 1]] = 0
template_refined = template_refined.reshape([dim[0], dim[1], dim[2]])

# Ensure template labels do not have gaps in the numbers, e.g. 0 1 3 ...
rois = np.unique(template_refined)
template_refined = (rois[:, np.newaxis, np.newaxis, np.newaxis] == template_refined[np.newaxis, :]).astype(int).argmax(0)

# Saving the template_refined
io.savemat(os.path.join(BASE_DIR, "group/ica_roi_parcel500_refined.mat"), {"template": template_refined})
nii = nib.Nifti1Image(template_refined, brain_img.affine)
nib.save(nii, os.path.join(BASE_DIR, "group/ica_roi_parcel500_refined.nii"))

            

