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

def readBvecsRow(bvecFile):
    lines=bvecFile.readlines()
    xvec=lines[0].split()
    yvec=lines[1].split()
    zvec=lines[2].split()
    for num in range(len(xvec)):
        xvec[num]=float(xvec[num])
        yvec[num]=float(yvec[num])
        zvec[num]=float(zvec[num])
    return (numpy.array(xvec), numpy.array(yvec), numpy.array(zvec))

def readBvecsCol(bvecFileName):
    with open(bvecFileName) as f:
        lines=f.readlines()
    xvec = [float(line.split()[0]) for line in lines]
    yvec = [float(line.split()[1]) for line in lines]
    zvec = [float(line.split()[2]) for line in lines]
    return (numpy.array(xvec), numpy.array(yvec), numpy.array(zvec))

def getBVecs(bvalFileName, bvecFileName, nifti):
    image=nib.load(nifti)
    data=image.get_header()    
    transform=data['pixdim'][1:4]/(sum(data['pixdim'][1:4]**2))**0.5
    length=data.get_data_shape()[3]
    bvecs=numpy.zeros(length*3)
    bvals=numpy.zeros(length)

    bvals = []
    with open(bvalFileName) as f:
        bvals = [float(line.strip()) for line in f.readlines()]

    (xvec, yvec, zvec) = readBvecsCol(bvecFileName)
    #(xvec, yvec, zvec) = readBvecsRow(bvecFileName)
    
    count=1
    for num in range(len(xvec)):
        newLine=numpy.array([xvec[num], yvec[num], zvec[num]])
        magnitude=sum(newLine**2)
        if magnitude == 0:
            magnitude = 1
        elif magnitude <0.95 or magnitude >1.05:
            print "WARNING: The following vector in this file may not be unit vectors. A vector was found with a magnitude straying more than 5% from 1"
            print "magnitude: %s" % str(magnitude)            
            print "vector: %s" % str(newLine)
            print "vector number %s" % str(count)
            count = count+1
        print transform
        bvecs[(num*3):((num+1)*3)]=numpy.array((newLine/(magnitude**.5))[0:3]*(bvals[num]/max(bvals))**0.5)*transform
    print bvecs
    #bvecs[(num*3):((num+1)*3)]=(newLine/(magnitude**0.5))[0:3]*(bvals[num]/bval)**0.5
    bvecs=bvecs.reshape([length,3])
    return bvecs, max(bvals)


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
    centerings='cell cell cell ???'
    kinds= 'space space space list'
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
             'space directions: %s' % coordinates + 'none', 
             'centerings: %s' % centerings, 
             'kinds: %s' % kinds, 
             'endian: %s' % endian, 
             'encoding: %s' % encoding, 
             'space units: %s' % space_units,
             'space origin: %s' % origin,
             'measurement frame: %s' % frame]

    return header


def createNrrd(nifti, header, bval, bvecs):
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
    nrrd.write('\nmodality:=DWMRI')
    nrrd.write('\nDWMRI_b-value:=' + str(bval))
    for num in range(bvecs.size/3):
        row= str(bvecs[num])
        row=re.sub('\[', '', row, count=0)
        row=re.sub('\]', '', row, count=0)
        zeroNum='%04d' % num
        nrrd.write('\nDWMRI_gradient_%s' % zeroNum + ':= %s' % row)
    nrrd.write('\n')
    nrrd.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Enter location of DWI, bval, and bvec files')
    parser.add_argument('-i', dest='dwi_file')
    parser.add_argument('-bval', dest='bval_file')
    parser.add_argument('-bvec', dest='bvec_file')
    dwi_file = parser.parse_args().dwi_file
    bval_file = parser.parse_args().bval_file
    bvec_file = parser.parse_args().bvec_file
    [bvecs, bval] = getBVecs(bval_file, bvec_file, dwi_file)
    header = makeHeader(dwi_file)
    createNrrd(dwi_file, header, bval, bvecs)
