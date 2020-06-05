#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downloads MNIST dataset and creates an HDF5 for 'train', 'validation' and 'test' sets
"""

import urllib.request
import pickle
import gzip
import h5py
import os 

# make mnist directory
try:
    os.mkdir('../data/mnist/')
except OSError:
    pass
        

# download MNIST
ulr_mnist = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
mnist_filename = 'mnist.pkl.gz'
urllib.request.urlretrieve(ulr_mnist, '../data/mnist/' + mnist_filename)
      
# 'mnist.pkl.gz' was created in Python2, 
#  thus  'latin1' encoding is needed in read it in Python3
with gzip.open('../data/mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# Save each set as an individual HDF5 file     
datasets = [train_set, valid_set, test_set, test_set]
dataset_names = ['train', 'validation', 'test']

for dataset, dataset_name in zip(datasets, dataset_names):
    print(dataset_name)  
    file = h5py.File('../data/mnist/' + dataset_name + '.hdf5', 'w')
    file.create_dataset('samples', data=dataset[0])
    file.create_dataset('labels' , data=dataset[1])
    file.close()