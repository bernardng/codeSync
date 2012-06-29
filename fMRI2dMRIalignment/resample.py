"""
Resample input volume to resolution of reference
"""
import os
import nibabel as nib
import numpy as np
import argparse
from nipy.labs import as_volume_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resample input volume to resolution of reference (-t 0 = continuous interpolation; -t 1 = nearest neighbor')
    parser.add_argument('-i', dest='input')
    parser.add_argument('-r', dest='reference')
    parser.add_argument('-t', dest='interp_type', default=0)
    input_path = parser.parse_args().input
    ref_path = parser.parse_args().reference
    interp_type = np.float32(parser.parse_args().interp_type)
    if interp_type == 0:
	interp = 'continuous'
    elif interp_type == 1:
	interp = 'nearest'
    path, afile = os.path.split(input_path)
    afile = afile.replace('.nii', '_rs.nii')
    vol_img = as_volume_img(input_path)
    ref_img = as_volume_img(ref_path)
    vol_img = vol_img.resampled_to_img(ref_img, interpolation=interp)
    vol = vol_img.get_data()
    nii = nib.Nifti1Image(vol, ref_img.get_affine())
    nib.save(nii, os.path.join(path, afile))
