"""
Preprocessing 
Input:  tc_img = volume image of timecourses
        gm_img = volume image of gray matter mask
        wm_img = volume image of white matter mask
        csf_img = volume image of CSF
        motion_confounds = 6 rigid transform parameters during motion correction
        template_img = template to extract ROI timecourse
        tr = repetition time
Output: tc = preprocessed voxel time courses
        tc_roi = preprocessed ROI time courses
        tissue_mask = 1:GM, 2:WM, 3:CSF, for extracting GM voxels
"""
import pylab as pl
import numpy as np
import parietal
from parietal.time_series import preprocessing
from scipy import stats
from scipy import signal
from scipy import linalg
from nipy.labs import mask as mask_utils

def preproc(tc_img, gm_img, wm_img, csf_img, motion_confounds, task_regressors=None, tr=2.0, template_img=None):
    # Extract timeseries
    tc = tc_img.get_data()
    n_tpts = tc.shape[3]    
    tc = tc.T.reshape((n_tpts, -1))
    # Remove NAN's
    tc[np.isnan(tc)] = 0
    # Standardize motion regressors
    motion_confounds = preprocessing.standardize(motion_confounds.T).T
    
    # Temporal detrending
    if task_regressors: # Discrete cosine for task data
        nCond = task_regressors.shape[1]        
        regressors = np.hstack((task_regressors, motion_confounds))
        T = tr * n_tpts
        T_cut = 128 # Default for SPM is 128 sec
        t = np.arange(0, tr * n_tpts, tr)
        for i in np.arange(np.floor(2 * T / T_cut)):
            regressors = np.hstack((regressors, signal.cos(i * np.pi * t / T))
        beta, _, _, _ = linalg.lstsq(regressors, tc)        
        tc -= np.dot(regressors[:,0:nCond-1], beta[0:ncond-1,:])        
    else: # Resting state
        # Remove motion artifacts        
        beta, _, _, _ = linalg.lstsq(motion_confounds,tc)
        tc -= np.dot(motion_confounds, beta)        
        # Bandpass filter to remove DC offset and low frequency drifts
        f_cut = np.array([0.01, 0.1])
        samp_freq = 1 / tr 
        w_cut = f_cut * 2 / samp_freq
        b, a = signal.butter(5, w_cut, btype='bandpass', analog=0, output='ba')
        for temp in tc.T: # Modify tc in place
            temp[:] = signal.filtfilt(b, a, temp)

    # Resample tissue masks
    gm_img = gm_img.resampled_to_img(tc_img)
    wm_img = wm_img.resampled_to_img(tc_img)
    csf_img = csf_img.resampled_to_img(tc_img)
    
    # Extract tissue mask
    # Notes: Since tc transposed to speed up reshape, all masks below have to be transposed
    gm = gm_img.get_data().T 
    wm = wm_img.get_data().T
    csf = csf_img.get_data().T
    probTotal = gm + wm + csf
    ind = probTotal > 0
    gm[ind] = gm[ind] / probTotal[ind]
    wm[ind] = wm[ind] / probTotal[ind]
    csf[ind] = csf[ind] / probTotal[ind]
    tissue = np.array([gm.ravel(), wm.ravel(), csf.ravel()])
    tissue_mask = tissue.argmax(axis=0) + 1
    tissue_mask[probTotal.ravel() == 0] = 0

    # Remove WM and CSF signal from GM voxels
    tc_gm = tc[:, tissue_mask == 1] # Modifies tc in place
    tc_wm = np.mean(tc[:, tissue_mask == 2], axis=1)
    tc_csf = np.mean(tc[:, tissue_mask == 3], axis=1)
    beta, _, _, _ = linalg.lstsq(np.array([tc_wm, tc_csf]).T, tc_gm)
    tc_gm -= np.dot(np.array([tc_wm, tc_csf]).T, beta) 
    
    # Standardize the timecourses
    tc = preprocessing.standardize(tc.T).T    

    # Extract ROI timecourses
    if template_img:
        template = template_img.get_data().T.ravel()
        label = np.unique(template)    
        tc_roi = np.zeros((n_tpts, label.shape[0]))
        n_vox = np.zeros((label.shape[0]-1))
        for i in np.arange(label.shape[0]-1):
            # Find GM voxels within ROI        
            ind = (template == label[i+1]) & (tissue_mask == 1)
            n_vox[i] = np.sum(ind)            
            tc_roi[:, i] = np.mean(tc[:, ind], axis=1)
        tc_roi = preprocessing.standardize(tc_roi.T).T
    else:
        tc_roi = []
   
    return tc, tc_roi, tissue_mask  
    

