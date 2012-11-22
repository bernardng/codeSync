import os
import argparse
import numpy as np
import nibabel as nib
from nipy.labs import as_volume_img
from scipy.stats import scoreatpercentile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Normalize DWI intensity for group fiber tractography.')
    parser.add_argument('-i', dest='dwi_file') # 4D DWI Nifti
    dwi_path = parser.parse_args().dwi_file
    [output_path, dwi_file] = os.path.split(dwi_path)
    dwi_file = dwi_file.replace(".nii", "_norm.nii")
    dwi_img = as_volume_img(dwi_path)
    dwi = dwi_img.get_data()
    dwi = dwi / np.float(scoreatpercentile(dwi.ravel(), 99))
    dwi = np.int16(dwi * 1e4)   
    nii = nib.Nifti1Image(dwi, dwi_img.affine)
    nib.save(nii, os.path.join(output_path, dwi_file))
    
