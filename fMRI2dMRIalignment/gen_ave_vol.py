"""
Create average volume for generating brain mask and aligning fMRI to dMRI
"""
import os
import nibabel as nib
import numpy as np
import argparse
from nipy.labs import as_volume_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average of a 4D volume.')
    parser.add_argument('-i', dest='filename') # Takes .nii as input
    filepath = parser.parse_args().filename
    path, afile = os.path.split(filepath)
    afile = afile.replace('.nii', '_ave.nii')
    vol_img = as_volume_img(filepath)
    vol = vol_img.get_data()
    mean_vol = np.mean(vol, 3)
    nii = nib.Nifti1Image(mean_vol, vol_img.get_affine())
    nib.save(nii, os.path.join(path, afile))
    

