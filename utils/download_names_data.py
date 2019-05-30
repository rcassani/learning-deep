#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downloads a directory are 18 text files named as “[Language].txt”. 
Each file contains a bunch of names, one name per line, 
mostly romanized (but we still need to convert from Unicode to ASCII).

Data is available at https://download.pytorch.org/tutorial/data.zip
"""

import urllib.request
import zipfile
import os 

# make mnist directory
try:
    os.mkdir('../data/names/')
except OSError:
    pass
        

# download MNIST
ulr_names_pytorch = 'https://download.pytorch.org/tutorial/data.zip'
names_filename = 'data.zip'
urllib.request.urlretrieve(ulr_names_pytorch, '../data/names/' + names_filename)

zip_ref = zipfile.ZipFile('../data/names/' + names_filename, 'r')

for file_names in zip_ref.namelist():
    if file_names.startswith('data/names/'):
        zip_ref.extract(file_names, '../' )
