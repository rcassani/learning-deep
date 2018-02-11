import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, hdf5_name, ram=False):
        super(Loader, self).__init__()
        self.ram = ram
        self.hdf5_name = hdf5_name
        open_file = h5py.File(self.hdf5_name, 'r')
        if self.ram:
            self.labels = open_file['labels'][()]
            self.samples = open_file['samples'][()]
            self.__len = len(self.labels)
        else:
            labels = open_file['labels']
            self.__len = len(labels)
        open_file.close()		 
    
    def __getitem__(self, index):
        if self.ram:
            return torch.from_numpy(self.samples[index]).float(), torch.from_numpy(np.array([self.labels[index]])).long() 
        else:
            open_file = h5py.File(self.hdf5_name, 'r')  
            sample = open_file['samples'][index]
            label = open_file['labels'][index]
            open_file.close()		
        return torch.from_numpy(sample).float(), torch.from_numpy(np.array([label])).long()

    def __len__(self):
        return self.__len
    