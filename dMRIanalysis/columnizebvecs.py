import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert gradient table from 3 rows format to 3 columns.')
    parser.add_argument('-i', dest='filename')
    filepath = parser.parse_args().filename
    path, afile = os.path.split(filepath)
    afile = afile.replace('.bvec', '.txt')
    bvec = np.loadtxt(filepath).T
    np.savetxt(os.path.join(path, afile), bvec, fmt='%1.6f')
    
