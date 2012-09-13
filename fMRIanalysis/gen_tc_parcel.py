"""
Generate parcel timecourses for IMAGEN data
Input:  tc_file = location of preprocessed time courses (.mat)
        gm_file = location of gray matter mask (.nii)
        wm_file = location of white matter mask (.nii)
        csf_file = location of CSF mask (.nii)
        template_file = location of parcel template (.nii)
Output: tc_parcel = parcel time courses saved as tc_parcel.mat to folder of tc_file
"""
import os
import numpy as np
import argparse
from nipy.labs import as_volume_img
from scipy import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate parcel timecourses.')
    parser.add_argument('-tc', dest='tc_file')
    parser.add_argument('-gm', dest='gm_file')
    parser.add_argument('-wm', dest='wm_file')
    parser.add_argument('-csf', dest='csf_file')
    parser.add_argument('-t', dest='template_file')
    tc_file = parser.parse_args().tc_file
    gm_file = parser.parse_args().gm_file
    wm_file = parser.parse_args().wm_file
    csf_file = parser.parse_args().csf_file
    template_file = parser.parse_args().template_file
    path, afile = os.path.split(tc_file)
    _, template_name = os.path.split(template_file)
    template_name = template_name.replace('.nii', '')

    # Load parcel template
    template_img = as_volume_img(template_file)
    template = template_img.get_data().ravel()
    label = np.unique(template)

    # Load preprocessed voxel timecourses    
    tc = io.loadmat(tc_file)
    tc = tc["tc"]
    
    # Generate tissue mask
    gm_img = as_volume_img(gm_file)
    gm_img = gm_img.resampled_to_img(template_img)
    gm = gm_img.get_data()
    gm[gm < 0] = 0
    wm_img = as_volume_img(wm_file)
    wm_img = wm_img.resampled_to_img(template_img)
    wm = wm_img.get_data()
    wm[wm < 0] = 0
    csf_img = as_volume_img(csf_file)
    csf_img = csf_img.resampled_to_img(template_img)
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

    # Generate parcel timecourses   
    tc_parcel = np.zeros((tc.shape[0], label.shape[0] - 1))
    for i in np.arange(label.shape[0] - 1): # Skipping background
        ind = (template == label[i + 1]) & (tissue_mask == 1)
        tc_parcel[:, i] = np.mean(tc[:, ind], axis=1)

    # Save parcel timecourses
    io.savemat(os.path.join(path, "tc_" + template_name + ".mat"), {"tc": tc_parcel})

