#!/usr/bin/env python

#TODO

#1) Add measurement frame
#2) Save data part of nifti to its own file, and point to it in 'data file' field


import nibabel as nib
import sys
import numpy
import re
import os
import gzip
import argparse

def makeHeader(nifti):
    
    image=nib.load(nifti)
    data=image.get_header()
    pixdim=data['pixdim'][1:4]
    sizes=str(data.get_data_shape())
    sizes=re.sub('\(', '', sizes, count=0)
    sizes=re.sub('\)', '', sizes, count=0)
    sizes=re.sub('\,', '', sizes, count=3)
    dimensions=len(data.get_data_shape())
    space='RAS'
    centerings='cell cell cell'
    kinds= 'space space space'
    endian= 'little'
    encoding= 'gzip'
    frame= '(1,0,0) (0,1,0) (0,0,1)'
    space_units= '"mm" "mm" "mm"'
    coordinates= '('+ str(pixdim[0])+', 0.0, 0.0)'+' (0.0, '+str(pixdim[1])+', 0.0)'+' (0.0, 0.0, '+str(pixdim[2])+') '

    origin= str(list(data.get_sform()[0:3,3]))
    origin=re.sub('\[', '(', origin, count=0)
    origin=re.sub('\]', ')', origin, count=0)
    dtype= data.get_data_dtype()
    name=nifti.replace('.nii.gz','.nrrd')
    print numpy.array(coordinates)

    header= ['NRRD0005\n# Complete NRRD file format specification at:', 
             '# http://teem.sourceforge.net/nrrd/format.html', 
             'type: %s' % dtype, 
             'dimension: %d' % dimensions, 
             'space: %s' % space, 
             'sizes: %s' % sizes, 
             'space directions: %s' % coordinates, 
             'centerings: %s' % centerings, 
             'kinds: %s' % kinds, 
             'endian: %s' % endian, 
             'encoding: %s' % encoding, 
             'space units: %s' % space_units,
             'space origin: %s' % origin,
             'measurement frame: %s' % frame]

    return header


def createNrrd(nifti, header):
    name=nifti.replace('.nii','.nhdr')
    nrrd=open(name, 'w')
    nrrd.write('\n'.join(header))
    image=nib.load(nifti)
    [path, afile] =os.path.split(nifti)
    afile=afile.replace('.nii','.raw.gz')
    dataFileName = os.path.join(path, afile)
    dataFile=gzip.open(dataFileName,'wb')
    dataFile.write(image.get_data())
    dataFile.close()
    nrrd.write('\ndata file: ' + afile)
    nrrd.write('\n')
    nrrd.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input', dest='files', nargs='+', required=True, help='enter the name of the nifti files to be converted') # Takes .nii as input
    args = parser.parse_args()

    for nifti in args.files:
        header = makeHeader(nifti)
        createNrrd(nifti, header)
