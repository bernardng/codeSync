import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert rows in input file to columns.')
    parser.add_argument('-i', dest='input_file') # e.g. bvec.bvec with three rows of [gx;gy;gz] or bval.bval with a row of b-values
    parser.add_argument('-o', dest='output_file')
    parser.add_argument('-d', dest='num_dec')
    input_path = parser.parse_args().input_file
    output_path = parser.parse_args().output_file
    _, ext = os.path.splitext(input_path)
    x = np.loadtxt(input_path).T
    dim = x.shape
    if ext == ".txt" and np.size(dim) == 1: # if input is .txt, the output should be .bval with a row of b-values
        x = x[:, np.newaxis]
        x = x.T
    num_dec = parser.parse_args().num_dec
    np.savetxt(output_path, x, fmt='%1.' + num_dec + 'f')
    
