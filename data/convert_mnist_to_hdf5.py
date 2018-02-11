#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converts a MNIST datset from pickle format to HDF5
"""

import pickle
import gzip
import h5py

# As 'mnist.pkl.gz' was created in Python2, 'latin1' encoding is needed to loaded in Python3
with gzip.open('./mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# Save each set as an individual HDF5 file     
datasets = [train_set, valid_set, test_set, test_set]
dataset_names = ['train', 'validation', 'test', '']

for dataset, dataset_name in zip(datasets, dataset_names):
    print(dataset_name)  
    h5file = h5py.File('./mnist_' + dataset_name + '.hdf5', 'w')
    h5file.create_dataset('samples', data=dataset[0])
    h5file.create_dataset('labels' , data=dataset[1])
    h5file.close
