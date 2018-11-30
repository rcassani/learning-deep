"""
Loaders for CIFAR-10 dataset
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets as ptdsets

class Loader(Dataset):
    def __init__(self, set_type=None):
        super(Loader, self).__init__()
        # CIFAR-10 dataset #50K Training, #10K Test
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        if set_type == 'train' or set_type == 'validation':         
            trainset = ptdsets.CIFAR10(root='./data', download=True, train=True)
            # Split Training into Training (40K) and Validation(10K)
            y = np.array(trainset.train_labels)
            X = trainset.train_data
            ixt = np.reshape(np.argsort(y), [10, len(y)//10])
            if set_type == 'train':
                ix = ixt[:, 0:4000].flatten('C')    
            if set_type == 'validation':
                ix = ixt[:, 4000:].flatten('C')
            # Shuffle indices
            np.random.shuffle(ix)
            y = y[ix]
            X = X[ix, :, :, :]
              
        elif set_type == 'test':
            # 10k samples for testing
            testset = ptdsets.CIFAR10(root='./data', download=True, train=False)
            y = np.array(testset.test_labels)
            X = testset.test_data
        else:
            print("Error, value for 'set_type' parameter is not valid")
            return
            
        self.labels = y
        # Move X axes to be [example, channel, height, width]
        X = np.moveaxis(X, [0, 1, 2, 3], [0, 2, 3, 1])    
        self.samples = X
        self.__len = len(self.labels)
        
    def __getitem__(self, index):
            return torch.from_numpy(self.samples[index, :, :, :]).float(), torch.from_numpy(np.array([self.labels[index]])).long() 
       
    def __len__(self):
        return self.__len