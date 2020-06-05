#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downloads MRI data and mask for CorpusCallosum and prepares it.
Data is located in the IntroToDeepLearning repository by Robert Brown
https://github.com/robb-brown/IntroToDeepLearning
"""

import os 
import glob
import urllib.request
import nilearn as nilearn
import tarfile
import h5py
import numpy as np
from matplotlib import pyplot as plt

# make cc_data directory
cc_data_dir = '../data/corpus_callosum/'
try:
    os.mkdir(cc_data_dir)
except OSError:
    pass
        

# download data
ulr_ccdata = 'https://github.com/robb-brown/IntroToDeepLearning/raw/master/6_MRISegmentation/corpusCallosum.tar.gz'
ccdata_filename = 'corpusCallosum.tar.gz'
urllib.request.urlretrieve(ulr_ccdata, cc_data_dir + ccdata_filename)

# extract files
tar = tarfile.open(cc_data_dir + ccdata_filename, "r:gz")
tar.extractall(cc_data_dir)
tar.close()

X_list = []
Y_list = []

# for all the file pairs
for file in glob.glob( cc_data_dir + '?????.nii.gz'):
  ni_file = file
  cc_file = file.replace('.nii.gz','_cc.nii.gz')

  ni = nilearn.image.get_data(ni_file).transpose()
  cc = nilearn.image.get_data(cc_file).transpose()

  # each ni_file contains 3 sagital slices
  # these are saved independently for data augmentation
  for ix_slide in range(ni.shape[2]):
    # MRI slices
    ni_slice = np.expand_dims(ni[::-1, :, ix_slide], 2)
    X_list.append(ni_slice)
    # Segmentation data is located in the slice with index [1] of cc 
    cc_slice = np.expand_dims(cc[::-1, :, 1], 2)
    Y_list.append(cc_slice)

# stack all the MRI slices and their segmentations
X = np.stack(X_list, axis=-1)
Y = np.stack(Y_list, axis=-1)

hdf5file = h5py.File(cc_data_dir + 'corpus_callosum.hdf5', 'w')
hdf5file.create_dataset('X', data=X)
hdf5file.create_dataset('Y', data=Y)
hdf5file.close()
  
#plt.figure()
#plt.subplot(1,3,1)
#plt.imshow(X[:,:,0,0])
#plt.contour(Y[:,:,0,0], alpha=1)
#plt.subplot(1,3,2)
#plt.imshow(X[:,:,0,1])
#plt.contour(Y[:,:,0,1], alpha=1)
#plt.subplot(1,3,3)
#plt.imshow(X[:,:,0,2])
#plt.contour(Y[:,:,0,2], alpha=1)









