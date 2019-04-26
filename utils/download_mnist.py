#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downloads MNIST dataset and creates an HDF5 for 'train', 'validation' and 'test' sets
"""

import urllib.request
import pickle as pickle
import gzip
import os
import numpy as np


path = 'http://deeplearning.net/data/mnist'
mnist_filename_all = 'mnist.pkl'
local_filename = os.path.join(args.savedir, mnist_filename_all)
urllib.request.urlretrieve(
    "{}/{}.gz".format(path,mnist_filename_all), local_filename+'.gz')
tr,va,te = pickle.load(gzip.open(local_filename+'.gz','r'), encoding='latin1')
np.save(open(local_filename+'.npy','w'), (tr,va,te))
        
        
