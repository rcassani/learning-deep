"""
Loaders for MNIST dataset
Data is kept in RAM
"""

import torch
import pickle, gzip
import numpy as np
import os 
from torch.utils.data import Dataset


class Loader(Dataset):
    def __init__(self, set_type=None):
        super(Loader, self).__init__()
        # Import MNIST data
        dir_this_script = os.path.dirname(os.path.realpath(__file__))
        with gzip.open(dir_this_script  + '/../data/mnist/mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        if set_type == 'train':         
            # 50K samples for training
            X = train_set[0]
            y = train_set[1]           
        elif set_type == 'validation':
            # 10K samples for validation
            X = valid_set[0]
            y = valid_set[1]               
        elif set_type == 'test':
            # 10k samples for testing
            X = test_set[0]
            y = test_set[1]                   
        else:
            print("Error, value for 'set_type' parameter is not valid")
            return
            
        self.labels = y
        self.samples = X
        self.__len = len(self.labels)
        
    def __getitem__(self, index):
            return torch.from_numpy(self.samples[index]).float(), torch.from_numpy(np.array([self.labels[index]])).long() 
       
    def __len__(self):
        return self.__len