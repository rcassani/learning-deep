#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Downloads data for the Dogs vs Cats InclassKaggle challenge 
for image classification. The challege was part of the assignment #1 for the 
course IFT 6135 Representation Learning Winter 2019
https://sites.google.com/mila.quebec/ift6135.

Data is available at https://www.kaggle.com/c/ift6135h19/data
"""

import webbrowser
import os 
import zipfile

# make directory
try:
    os.mkdir('../data/dogs_vs_cats/')
except OSError:
    pass

# Download the files 'sample_submission.csv', 'testset.zip', 'trainset.zip'
webbrowser.open('https://www.kaggle.com/c/ift6135h19/data', new=2)


# Unzip if the files are already downloaded
for file_name in ['testset.zip', 'trainset.zip']:
  with zipfile.ZipFile('../data/dogs_vs_cats/' + file_name, 'r') as zip_ref:
    zip_ref.extractall('../data/dogs_vs_cats/')


        
