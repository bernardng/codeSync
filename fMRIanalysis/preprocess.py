"""
Preprocessing 
Input:  tc_file = location of time courses to be preprocessed (.nii)
        gm_file = location of gray matter mask (.nii)
        wm_file = location of white matter mask (.nii)
        csf_file = location of CSF mask (.nii)
        reg_file = location of regressor matrix (.txt/.mat)
        dtype = task (0) or resting state (1) data
        TR = repetition time
        n_cond = #experimental conditions, i.e. #task regressors in regressor matrix
Output: tc = preprocessed voxel time courses saved as tc_vox.mat to folder of tc_file
"""
import os
import numpy as np
import argparse
from nipy.labs import as_volume_img
from scipy import io
from scipy import signal
from scipy import linalg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal preprocessing.')
    parser.add_argument('-tc', dest='tc_file')
    parser.add_argument('-gm', dest='gm_file')
    parser.add_argument('-wm', dest='wm_file')
    parser.add_argument('-csf', dest='csf_file')
    parser.add_argument('-reg', dest='reg_file')
    parser.add_argument('-dtype', dest='dtype')
    parser.add_argument('-tr', dest='tr')
    parser.add_argument('-c', dest='n_cond')    
    tc_file = parser.parse_args().tc_file
    gm_file = parser.parse_args().gm_file
    wm_file = parser.parse_args().wm_file
    csf_file = parser.parse_args().csf_file
    reg_file = parser.parse_args().reg_file
    data_type = np.float32(parser.parse_args().dtype)
    TR = np.float32(parser.parse_args().tr)
    if data_type == 0:
        n_cond = np.float32(parser.parse_args().n_cond)
    path, afile = os.path.split(tc_file)
    
    if data_type:  
        # Rest data
        motion_confounds = np.loadtxt(reg_file) # Contains motion regressors
        regressors = motion_confounds.copy()
        # Add shifted motion regressors
        for i in np.array([-1, 1]):
            regressors = np.hstack((regressors, np.roll(motion_confounds, i, axis=0)))    
    else:
        regressors = io.loadmat(reg_file)
        regressors = regressors['SPM'][0, 0].xX[0, 0].X # Contains task and SHIFTED versions of motion regressors
#        # Data from other than IMAGEN database might not have shifted motion regressors
#        motion_confounds = regressors[:, n_cond:]
#        for i in np.array([-1, 1]):
#            regressors = np.hstack((regressors, np.roll(motion_confounds, i, axis=0)))
    
    print "Extracting time series..."        
    # Extract timeseries
    tc_img = as_volume_img(tc_file)
    tc = tc_img.get_data()
    tc_dim = tc.shape
    n_tpts = tc_dim[3]    
    tc = tc.reshape((-1, n_tpts)).T
    tc[np.isnan(tc)] = 0 # Remove NAN's
    
    print "Resampling tissue masks to EPI resolution..."
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
    if data_type == 1: # Resting state
        print "Removing confounds..."            
        # Standardize all regressors
        reg_std = np.std(regressors, axis=0) 
        ind = reg_std > 1e-16 # Avoid div by 0 when regressors contain a column of ones
        regressors[:, ind] = regressors[:, ind] - np.mean(regressors[:, ind], axis=0)
        regressors[:, ind] = regressors[:, ind] / reg_std[ind]
        if np.sum(ind) == regressors.shape[1]: # If all regressors have non-zero std, then insert a column of ones
            regressors = np.hstack((regressors, np.ones((n_tpts, 1)))) # a column of ones
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
        for i in np.arange(np.floor(2 * T / T_cut)) + 1: 
            regressors = np.hstack((regressors, signal.cos(i * np.pi * t / T).reshape(-1, 1)))
        # Standardize all regressors
        reg_std = np.std(regressors, axis=0) 
        ind = reg_std  > 1e-16 # Avoid div by 0 when regressors contain a column of ones
        regressors[:, ind] = regressors[:, ind] - np.mean(regressors[:, ind], axis=0)
        regressors[:, ind] = regressors[:, ind] / reg_std[ind]
        if np.sum(ind) == regressors.shape[1]: # If all regressors have non-zero std, then insert a column of ones
            regressors = np.hstack((regressors, np.ones((n_tpts, 1)))) # a column of ones
        beta, _, _, _ = linalg.lstsq(regressors, tc)        
        tc -= np.dot(regressors[:, n_cond:], beta[n_cond:, :])       

    io.savemat(os.path.join(path, "tc_vox.mat"), {"tc": tc})        
        
