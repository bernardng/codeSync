"""
Functional Parcellation for IMAGEN data
Input:	n_parcels = #parcels to divide the brain
	TC_PATH = file path to tc_vox.mat
	REF_PATH = .nii in 3mm MNI space, e.g. RS-fMRI volumes
	GM_PATH = file path to probabilistic gray matter mask
	WM_PATH = file path to probabilistic white matter mask
	CSF_PATH = file path to probabilistic CSF mask
	PARCEL_PATH = file path to save the parcellation
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
n_parcels = 500.0

# Change path to files
TC_PATH = "tc_vox.mat"
REF_PATH = "wea000000459848s007a001.nii.gz" 
GM_PATH = "../Maps/wc1mprage000000459848.nii.gz"
WM_PATH = "../Maps/wc2mprage000000459848.nii.gz"
CSF_PATH = "../Maps/wc3mprage000000459848.nii.gz"
PARCEL_PATH = "parcel500.nii"

# Load subject timecourse
tc = io.loadmat(TC_PATH)
tc = tc["tc"].T
        
# Generate dilated GM mask
ref_img = as_volume_img(REF_PATH) # For resampling to 3mm
ref = ref_img.get_data()
gm_img = as_volume_img(GM_PATH)
gm = gm_img.get_data()
#wm_img = as_volume_img(WM_PATH)
#wm = wm_img.get_data()
#csf_img = as_volume_img(CSF_PATH)
#csf = csf_img.get_data()
#probTotal = gm + wm + csf
#dim = np.shape(probTotal)
#ind = probTotal > 0
#gm[ind] = gm[ind] / probTotal[ind]
#wm[ind] = wm[ind] / probTotal[ind]
#csf[ind] = csf[ind] / probTotal[ind]
#tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
#tissue_mask = tissue.argmax(axis=0) + 1
#tissue_mask[probTotal.ravel() == 0] = 0
#brain = np.reshape(tissue_mask == 1, (dim[0], dim[1], dim[2]))
brain = gm > 0.15
brain = binary_dilation(brain, structure=generate_binary_structure(3, 2))
brain_img = VolumeImg(brain, affine=gm_img.affine, world_space=None, interpolation='nearest')
brain_img = brain_img.resampled_to_img(ref_img, interpolation='nearest')
brain = brain_img.get_data()
brain = binary_closing(brain, structure=generate_binary_structure(3, 2))
labels, n_labels = label(brain) # Remove isolated voxels
for i in np.unique(labels)[2:]:
   brain[labels == i] = 0

# Spatial smoothing to encourage smooth parcels
dim = np.shape(brain)
tc = tc.reshape((dim[0], dim[1], dim[2], -1))
n_tpts = tc.shape[-1]
for t in np.arange(n_tpts):
    tc[:, :, :, t] = gaussian_filter(tc[:, :, :, t], sigma=1)
tc = tc.reshape((-1, n_tpts))
tc = tc[brain.ravel() == 1, :]

# Functional parcellation with Ward clustering
print("Performing Ward Clustering")
mem = Memory(cachedir='.', verbose=1)
# Define connectivity based on brain mask
A = grid_to_graph(n_x=brain.shape[0], n_y=brain.shape[1], n_z=brain.shape[2], mask=brain)
# Create ward object
ward = WardAgglomeration(n_clusters=n_parcels, connectivity=A.tolil(), memory=mem)
ward.fit(tc.T)
template = np.zeros((dim[0], dim[1], dim[2]))
template[brain==1] = ward.labels_ + 1 # labels start from 0, which is used for background

# Remove single voxels not connected to parcel
#for i in np.unique(template)[1:]:
#    labels, n_labels = label(template == i, structure=np.ones((3,3,3)))
#    if n_labels > 1:
#	for j in np.arange(n_labels):
#	    if np.sum(labels == j + 1) < 10:
#		template[labels == j + 1] = 0

# Saving the template
nii = nib.Nifti1Image(template, brain_img.affine)
nib.save(nii, PARCEL_PATH)
