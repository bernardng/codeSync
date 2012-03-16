"""
Preprocessing 
Input:  tc_img = volume image of timecourses
        gm_img = volume image of gray matter mask
        wm_img = volume image of white matter mask
        csf_img = volume image of CSF
        motion_confounds = 6 rigid transform parameters during motion correction
        template_img = template to extract ROI timecourse
        TR = repetition time
Output: tc = preprocessed voxel time courses
        tc_roi = preprocessed ROI time courses
"""
import os
import pylab as pl
import numpy as np
from parietal.time_series import preprocessing
from nipy.labs import as_volume_img
from nipy.labs.datasets.volumes.volume_img import VolumeImg
from nipy.labs import mask as mask_utils
from scipy import io
from scipy import stats
from scipy import signal
from scipy import linalg
import nibabel as nib

BASE_DIR = "/volatile/bernardng/data/imagen/"
TR = 2.2
#subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/subjectList.txt"), dtype='str')
subList = np.loadtxt(os.path.join(BASE_DIR, "subjectLists/facesList.txt"), dtype='str')

data_type = 0 # 0 = task, 1 = rest

# Loading Template
template_file = "/volatile/bernardng/templates/freesurfer/cort_subcort_333.nii"
template_img = as_volume_img(template_file)
template = template_img.get_data().ravel()    
for sub in subList:
    print str("Subject" + sub)
    # Define path to data
    gm_file = os.path.join(BASE_DIR, sub, "anat", "gmMask.nii")
    wm_file = os.path.join(BASE_DIR, sub, "anat", "wmMask.nii")
    csf_file = os.path.join(BASE_DIR, sub, "anat", "csfMask.nii")
    
    if data_type:  
        # Rest data
        tc_file = os.path.join(BASE_DIR, sub, "restfMRI", "rest.nii") 
        motion_confounds = np.loadtxt(os.path.join(BASE_DIR, sub, "restfMRI", "restSPM.txt")) # Contains motion regressors
        regressors = motion_confounds.copy()
        # Add shifted motion regressors
        for i in np.array([-1, 1]):
            regressors = np.hstack((regressors, np.roll(motion_confounds, i, axis=0)))    
    else:
#        # GCA data    
#        tc_file = os.path.join(BASE_DIR, sub, "gcafMRI", "gca.nii")    
#        regressors = io.loadmat(os.path.join(BASE_DIR, sub, "gcafMRI", "gcaSPM.mat"))
#        regressors = regressors['SPM'][0, 0].xX[0, 0].X # Contains task and SHIFTED versions of motion regressors
#        n_cond = 10 

        # faces data        
        tc_file = os.path.join(BASE_DIR, sub, "facesfMRI", "faces.nii")    
        regressors = io.loadmat(os.path.join(BASE_DIR, sub, "facesfMRI", "facesSPM.mat"))
        regressors = regressors['SPM'][0, 0].xX[0, 0].X # Contains task and SHIFTED versions of motion regressors
        regressors = regressors[:, np.arange(regressors.shape[1]) != 10] # removing neutral face regressor
        n_cond = 10 
    
    print "Extracting time series..."        
    # Extract timeseries
    tc_img = as_volume_img(tc_file)
    tc = tc_img.get_data()
    tc_dim = tc.shape
    n_tpts = tc_dim[3]    
    tc = tc.reshape((-1, n_tpts)).T
    tc[np.isnan(tc)] = 0 # Remove NAN's
    
    print "Downsampling tissue masks to EPI resolution..."
    # Load tissue data objects
    gm_img = as_volume_img(gm_file)
    gm_img = gm_img.resampled_to_img(tc_img)
    gm = gm_img.get_data()
    wm_img = as_volume_img(wm_file)
    wm_img = wm_img.resampled_to_img(tc_img)
    wm = wm_img.get_data()
    csf_img = as_volume_img(csf_file)
    csf_img = csf_img.resampled_to_img(tc_img)
    csf = csf_img.get_data()
    tissue_dim = gm.ravel().shape[0] # For checking if have to resample tissue to template
    
    print "Binarizing tissue masks..."
    # Binarizing tissue mask
    probTotal = gm + wm + csf
    ind = probTotal > 0
    gm[ind] = gm[ind] / probTotal[ind]
    wm[ind] = wm[ind] / probTotal[ind]
    csf[ind] = csf[ind] / probTotal[ind]
    tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
    tissue_mask = tissue.argmax(axis=0) + 1
    tissue_mask[probTotal.ravel() == 0] = 0

    print "Extracting WM and CSF confounds..."       
    # WM and CSF confounds
    tc_wm = np.mean(tc[:, tissue_mask == 2], axis=1)
    tc_csf = np.mean(tc[:, tissue_mask == 3], axis=1)
    confounds = np.array([tc_wm, tc_csf]).T    
    
    print "Extracting high variance voxel confounds"
    # High variance voxels
    std = np.std(tc, axis=0)
    ind = std.argsort()
    tc_highvar = tc[:, ind[::-1][:10]]       
    confounds = np.hstack((confounds, tc_highvar))
    
    # Adding shifted versions of wm, csf, and high variance voxels confounds    
    for i in np.array([-1, 0, 1]):
        regressors = np.hstack((regressors, np.roll(confounds, i, axis=0)))       

    # Temporal detrending
    if data_type: # Resting state
        print "Removing confounds..."            
        # Standardize all regressors
        regressors = preprocessing.standardize(regressors.T).T    
        regressors = np.hstack((regressors, np.ones((n_tpts, 1))))
        beta, _, _, _ = linalg.lstsq(regressors, tc)
        tc -= np.dot(regressors, beta)      
        
        # Bandpass filter to remove DC offset and low frequency drifts
        f_cut = np.array([0.01, 0.1])
        samp_freq = 1 / TR 
        w_cut = f_cut * 2 / samp_freq
        b, a = signal.butter(5, w_cut, btype='bandpass', analog=0, output='ba')
        for temp in tc.T: # Modify tc in place
            temp[:] = signal.filtfilt(b, a, temp)
    else: # Task
        T = TR * n_tpts
        T_cut = 128 # Default for SPM is 128 sec
        t = np.arange(0, TR * n_tpts, TR)
        for i in np.arange(np.floor(T / T_cut)) + 1: 
            regressors = np.hstack((regressors, signal.cos(i * np.pi * t / T).reshape(-1, 1)))
        # Standardize all regressors
        regressors = preprocessing.standardize(regressors.T).T            
        regressors = np.hstack((regressors, np.ones((n_tpts, 1)))) # a column of ones
        beta, _, _, _ = linalg.lstsq(regressors, tc)        
        tc -= np.dot(regressors[:, n_cond:], beta[n_cond:, :])       

#    if tissue_dim != template.shape[0]:
#        print "Resampling tissue masks to template resolution..."    
#        # Resample tissue mask to template resolution
#        gm_img = as_volume_img(gm_file)
#        gm_img = gm_img.resampled_to_img(template_img)
#        gm = gm_img.get_data()
#        wm_img = as_volume_img(wm_file)
#        wm_img = wm_img.resampled_to_img(template_img)
#        wm = wm_img.get_data()
#        csf_img = as_volume_img(csf_file)
#        csf_img = csf_img.resampled_to_img(template_img)
#        csf = csf_img.get_data()
#    
#        print "Binarizing resampled tissue masks..."
#        # Binarizing resampled tissue masks
#        probTotal = gm + wm + csf
#        ind = probTotal > 0
#        gm[ind] = gm[ind] / probTotal[ind]
#        wm[ind] = wm[ind] / probTotal[ind]
#        csf[ind] = csf[ind] / probTotal[ind]
#        tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
#        tissue_mask = tissue.argmax(axis=0) + 1
#        tissue_mask[probTotal.ravel() == 0] = 0
#        del gm, wm, csf, probTotal, tissue
#        del gm_img, wm_img, csf_img
#    
#    print "Extracting ROI timecourses..."    
#    # Extracting ROI timecourses
#    label = np.unique(template)    
#    n_vox = np.zeros((label.shape[0] - 1)) # For debugging
#    tc_roi = np.zeros((n_tpts, label.shape[0] - 1))
#    if tissue_dim != template.shape[0]:
#        tc = np.array(tc.T.reshape(tc_dim))
#        for j in np.arange(n_tpts):
#            tc_temp_img = VolumeImg(tc[:,:,:,j], tc_img.affine, world_space=None)
#            tc_temp_img = tc_temp_img.resampled_to_img(template_img)
#            tc_temp = tc_temp_img.get_data().ravel()
#            for i in np.arange(label.shape[0] - 1):
#                # Find GM voxels within ROI        
#                ind = (template == label[i + 1]) & (tissue_mask == 1)
#                n_vox[i] = np.sum(ind)      
#                print i
#                print n_vox[i]
#                tc_roi[j, i] = np.mean(tc_temp[ind])
#    else:
#        for i in np.arange(label.shape[0] - 1):
#            ind = (template == label[i + 1]) & (tissue_mask == 1)
#            n_vox[i] = np.sum(ind)
#            print i
#            print n_vox[i]
#            tc_roi[:, i] = np.mean(tc[:, ind], axis=1)
#    tc_roi = preprocessing.standardize(tc_roi.T).T
    
    if data_type:
#        io.savemat(os.path.join(BASE_DIR, sub, "restfMRI", "tc_rest_roi_fs.mat"), {"tc_roi": tc_roi})
        io.savemat(os.path.join(BASE_DIR, sub, "restfMRI", "tc_rest_vox.mat"), {"tc_vox": tc})
    else:
#        io.savemat(os.path.join(BASE_DIR, sub, "gcafMRI", "tc_task_roi_fs.mat"), {"tc_roi": tc_roi})
#        io.savemat(os.path.join(BASE_DIR, sub, "facesfMRI", "tc_task_roi_fs.mat"), {"tc_roi": tc_roi})
#        io.savemat(os.path.join(BASE_DIR, sub, "gcafMRI", "tc_task_vox.mat"), {"tc_vox": tc})
        io.savemat(os.path.join(BASE_DIR, sub, "facesfMRI", "tc_task_vox.mat"), {"tc_vox": tc})
        
        
