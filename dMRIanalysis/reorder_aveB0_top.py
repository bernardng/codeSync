import os
import argparse
import numpy as np
import nibabel as nib
from nipy.labs import as_volume_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put B0 volumes at the top.')
    parser.add_argument('-i', dest='dwi_file') # 4D DWI Nifti
    parser.add_argument('-b', dest='bval_file') # b-value txt file
    parser.add_argument('-d', dest='bvec_file') # gradient table
    dwi_path = parser.parse_args().dwi_file
    bval_path = parser.parse_args().bval_file
    bvec_path = parser.parse_args().bvec_file
    dwi_img = as_volume_img(dwi_path)
    dwi = dwi_img.get_data()
    bval = np.loadtxt(bval_path)
    bvec = np.loadtxt(bvec_path)
    
    b0_ave = np.mean(dwi[:, :, :, bval == 0], axis=3)
    dwi_reorder = np.concatenate((b0_ave[:, :, :, np.newaxis], dwi[:, :, :, bval != 0]), axis=3)
    bvec_reorder = np.array([0, 0, 0])
    bvec_reorder = np.vstack((bvec_reorder, bvec[bval != 0]))
    bval_reorder = np.array([0])
    bval_reorder = np.hstack((bval_reorder, bval[bval != 0]))
   
    nii = nib.Nifti1Image(dwi_reorder, dwi_img.affine)
    nib.save(nii, dwi_path)
    np.savetxt(bvec_path, bvec_reorder, fmt='%1.6f')
    np.savetxt(bval_path, bval_reorder, fmt='%1.0f')
