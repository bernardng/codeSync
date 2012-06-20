import os
import numpy as np
import argparse
import nibabel as nib
from nipy.labs import as_volume_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert data to uint8 (-t 0) or int16 (-t 1).')
    parser.add_argument('-i', dest='filename')
    parser.add_argument('-t', dest='filetype', type=int)
    filepath = parser.parse_args().filename
    input_type = parser.parse_args().filetype
    path, afile = os.path.split(filepath)
    vol_img = as_volume_img(filepath)
    if input_type == 0:
        afile = afile.replace('.nii', '_uint8.nii')
        vol = np.uint8(vol_img.get_data())
    elif input_type == 1:
        afile = afile.replace('.nii', '_int16.nii')
        vol = np.int16(vol_img.get_data())
    nii = nib.Nifti1Image(vol, vol_img.affine)
    nib.save(nii, os.path.join(path, afile))
