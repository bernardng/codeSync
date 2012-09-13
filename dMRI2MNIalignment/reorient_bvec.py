import os
import argparse
import numpy as np
import nibabel as nib
from nipy.labs import as_volume_img
from scipy.linalg import sqrtm
from scipy.linalg import inv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reorient b-vectors to account for affine registration.')
    parser.add_argument('-i', dest='bvec_file') # bvec.bvec, need to be 3 by #gradients
    parser.add_argument('-a', dest='aff_file') # affine transform txt file
    bvec_path = parser.parse_args().bvec_file
    [output_path, afile] = os.path.split(bvec_path)
    afile = afile.replace('.bvec', '_reorient.bvec')
    aff_path = parser.parse_args().aff_file
    bvec = np.loadtxt(bvec_path)
    aff = np.loadtxt(aff_path)
    aff = aff[:3, :3]
    R = np.dot(inv(sqrtm(np.dot(aff, aff.T))), aff) # Compute rotation matrix from affine transform based on Alexander et al., TMI 2001
    bvec = np.dot(R, bvec)
    np.savetxt(os.path.join(output_path, afile), bvec, fmt='%1.6f')

