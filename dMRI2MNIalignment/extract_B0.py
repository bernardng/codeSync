import os
import argparse
import numpy as np
import nibabel as nib
from nipy.labs import as_volume_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and average B0 volumes.')
    parser.add_argument('-i', dest='dwi_file') # 4D DWI Nifti
    parser.add_argument('-b', dest='bval_file') # b-value txt file
    dwi_path = parser.parse_args().dwi_file
    [output_path, afile] = os.path.split(dwi_path)
    bval_path = parser.parse_args().bval_file
    dwi_img = as_volume_img(dwi_path)
    dwi = dwi_img.get_data()
    bval = np.loadtxt(bval_path)
    b0 = np.mean(dwi[:, :, :, bval==0], axis=3)
    nii = nib.Nifti1Image(b0, dwi_img.affine)
    nib.save(nii, output_path + "b0.nii")
