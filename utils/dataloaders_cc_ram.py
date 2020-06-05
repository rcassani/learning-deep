"""
Loaders for Corpus callosum dataset
Data is kept in RAM
"""

import torch
import os 
from torch.utils.data import Dataset
import h5py

class Loader(Dataset):
    def __init__(self, set_type=None):
        super(Loader, self).__init__()
        # Import CC data
        dir_this_script = os.path.dirname(os.path.realpath(__file__))

        with h5py.File(dir_this_script  + '/../data/corpus_callosum/corpus_callosum.hdf5', 'rb') as f:
            # Get the data
            X_all = f['X']
            Y_all = f['Y']
        
        if set_type == 'train':         
            # 18 samples for training
            X = X_all[:,:,:,0:18]
            Y = Y_all[:,:,:,0:18]  
        elif set_type == 'validation':
            # 6 samples for validation
            X = X_all[:,:,:,18:24]
            Y = Y_all[:,:,:,18:24]  
        elif set_type == 'test':
            # 6 samples for testing
            X = X_all[:,:,:,24:30]
            Y = Y_all[:,:,:,24:30]  
        elif set_type == 'all':
            # 6 samples for testing
            X = X_all
            Y = Y_all                         
        else:
            print("Error, value for 'set_type' parameter is not valid")
            return
            
        self.Y = Y
        self.X = X
        self.__len = len(self.labels)
        
    def __getitem__(self, index):
            return torch.from_numpy(self.X[index]).float(), torch.from_numpy(self.Y[index]).long() 
       
    def __len__(self):
        return self.__len