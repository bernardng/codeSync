import os
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert rows in input file to columns.')
    parser.add_argument('-i', dest='input_file')
    parser.add_argument('-o', dest='output_file')
    parser.add_argument('-d', dest='num_dec')
    input_path = parser.parse_args().input_file
    output_path = parser.parse_args().output_file
    x = np.loadtxt(input_path).T
    num_dec = parser.parse_args().num_dec or '6'
    np.savetxt(output_path, x, fmt='%1.' + num_dec + 'f')
    
