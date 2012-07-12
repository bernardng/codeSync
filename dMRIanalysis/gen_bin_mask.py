import os
import numpy as np
import argparse
import nibabel as nib
from nipy.labs import as_volume_img
from scipy.ndimage.morphology import binary_dilation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarize tissue mask based on user-defined threshold (-t 0.33.')
    parser.add_argument('-i', dest='filename')
    parser.add_argument('-t', dest='threshold', default=0.33)
    filepath = parser.parse_args().filename
    thresh = np.float32(parser.parse_args().threshold)
    path, afile = os.path.split(filepath)
    afile = afile.replace('.nii', '_bin.nii')
    vol_img = as_volume_img(filepath)
    vol = vol_img.get_data() > thresh
    vol = binary_dilation(vol, structure=np.ones((3, 3, 3))).astype(np.uint8)
    nii = nib.Nifti1Image(vol, vol_img.affine)
    nib.save(nii, os.path.join(path, afile))

    
