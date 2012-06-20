"""
Functional Parcellation for IMAGEN data
"""
import os
import numpy as np
from scipy import io
from scipy.ndimage import gaussian_filter
from nipy.labs import as_volume_img
from nipy.labs import mask as mask_utils
from sklearn.decomposition import PCA
from sklearn.externals.joblib import Memory
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import WardAgglomeration
from parietal.time_series import preprocessing

BASE_DIR = "/volatile/bernardng/data/imagen/"
TR = 2.2
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectList.txt"), dtype='str')

# Concatenating PCA-ed voxel timecourses across subjects
for sub in subList:
    tc = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI/tc_rest_vox.mat"))
    tc = tc["tc_vox"]
    pca = PCA(n_components=10)    
    pca.fit(tc.T)
    if sub == subList[0]:
        tc_group = preprocessing.standardize(pca.transform(tc.T))
    else:
        tc_group = np.hstack((tc_group, preprocessing.standardize(pca.transform(tc.T))))
    print("Concatenating subject" + sub + "'s timecourses")
#io.savemat(os.path.join(BASE_DIR, "group/tc_rest_pca_vox.mat"), {"tc_group": tc_group})

# Perform parcellation on PCA-ed timecourses
brain_img = as_volume_img("/volatile/bernardng/templates/spm8/rgrey.nii")
brain = brain_img.get_data()
dim = np.shape(brain)
brain = brain > 0.2 # Generate brain mask
brain = mask_utils.largest_cc(brain)
mem = Memory(cachedir='.', verbose=1)
# Define connectivity based on brain mask
A = grid_to_graph(n_x=brain.shape[0], n_y=brain.shape[1], n_z=brain.shape[2], mask=brain)
# Create ward object
ward = WardAgglomeration(n_clusters=500, connectivity=A, memory=mem)
tc_group = tc_group.reshape((dim[0], dim[1], dim[2], -1))
n_tpts = tc_group.shape[-1]
for t in np.arange(n_tpts):
    tc_group[:,:,:,t] = gaussian_filter(tc_group[:,:,:,t], sigma=5)
tc_group = tc_group.reshape((-1, n_tpts))
tc_group = tc_group[brain.ravel()==1, :]
print("Performing Ward Clustering")
ward.fit(tc_group.T)
template = np.zeros((dim[0], dim[1], dim[2]))
template[brain==1] = ward.labels_ + 1 # Previously processed data did not include +1

# Remove parcels with zero timecourses in any of the subjects
template = template.ravel()
template_refined = template.copy()
label = np.unique(template)
for sub in subList:
    print str("Subject" + sub)
    # Load preprocessed voxel timecourses    
    tc = io.loadmat(os.path.join(BASE_DIR, sub, "restfMRI/tc_rest_vox.mat"))
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
    # Refine Parcellation    
    tc_parcel = np.zeros((tc.shape[0], label.shape[0] - 1))
    for i in np.arange(label.shape[0] - 1): # Skipping background
        ind = (template == label[i + 1]) & (tissue_mask == 1)
        tc_parcel[:, i] = np.mean(tc[:, ind], axis=1)
        if np.sum(tc_parcel[:, i]) == 0 or np.sum(ind) < 10:
            template_refined[template == label[i + 1]] = 0
template_refined = template_refined.reshape([dim[0], dim[1], dim[2]])
io.savemat(os.path.join(BASE_DIR, "group/parcel500refined.mat"), {"template": template_refined})
            

