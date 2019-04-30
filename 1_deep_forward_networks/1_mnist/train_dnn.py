#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:03:00 2017
@author: cassani
"""

import argparse
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import model_zoo                         # model to train 

# import scripts from the 'utils' directory 
import sys
sys.path.append('../../utils/')
from dataloaders_mnist_ram import Loader
from learning_loop import LearningLoop    # general learning loop

# Terminal: 
# python -i train_dnn.py --epochs=20 --ngpus=1 --lr=0.5 --l2=1 --model=mlp_notebook
# Spyder terminal
# runfile('train_dnn.py', '--epochs=20 --ngpus=1 --lr=0.5 --l2=1 --model=mlp_notebook')

# runfile('train_dnn.py', '--epochs=50 --ngpus=1 --lr=1 --l2=1 --model=mlp_notebook --train-batch-size=50000')


# Training settings
parser = argparse.ArgumentParser(description='Template PyTorch')
parser.add_argument('--train-batch-size', type=int, default=64,  metavar='N', help='input batch size for training (default:  64)')
parser.add_argument('--valid-batch-size', type=int, default=256, metavar='N', help='input batch size for testing  (default: 256)')
parser.add_argument('--test-batch-size' , type=int, default=256, metavar='N', help='input batch size for testing  (default: 256)')

parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 200)')

parser.add_argument('--model', type=str, default='mlp', metavar='N', help='name of the model, see model_zoo.py')

parser.add_argument('--patience', type=int, default=30, metavar='N', help='number of epochs without improvement to wait before stopping training (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--l2', type=float, default=0.00001, metavar='lambda', help='L2 wheight decay coefficient (default: 0.00001)')

parser.add_argument('--ngpus', type=int, default=0, help='Number of GPUs to use. Default=0 (no GPU)')

parser.add_argument('--checkpoint-path', type=str, default='./checkpoints', metavar='PATH', help='Directory for checkpoint files, if None or empty string, checkpoints are not saved')
parser.add_argument('--checkpoint-load', type=int, default=None, metavar='N', help='Indicates epoch to restart. If None, training starts from scratch')

parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--n-workers', type=int, default=1, metavar='N', help='Number of workers for dataloaders')

args = parser.parse_args()

# validate checkpoint-path input
if args.checkpoint_path == '' or args.checkpoint_path == 'None':
  args.checkpoint_path = None

# Verify CUDA
cuda_flag = args.ngpus > 0 and torch.cuda.is_available()
print('CUDA Mode is: ' +  str(cuda_flag))

# Random seed
torch.manual_seed(args.seed)
if cuda_flag:
    torch.cuda.manual_seed(args.seed)

# Loader will put all the dataset in RAM
ram = True

# Datasets and DataLoaders
train_dataset = Loader('train')
valid_dataset = Loader('validation')
test_dataset  = Loader('test')

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.n_workers)
valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers)
test_loader  = DataLoader(test_dataset , batch_size=args.test_batch_size , shuffle=True, num_workers=args.n_workers)

# 1. Model design and GPU capability
print('Model = ' + args.model)
model = getattr(model_zoo, args.model)()

# Load model in GPU(s)
if cuda_flag:
    if args.ngpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpus)))
    else:
        model = model.cuda()

# 2. Loss and Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr)
loss_funct = torch.nn.CrossEntropyLoss()

# 3. Create training loop and train
learner = LearningLoop(model, optimizer, loss_funct,
                       train_loader=train_loader, valid_loader=valid_loader,
                       checkpoint_path=args.checkpoint_path, checkpoint_load=args.checkpoint_load,
                       cuda_flag=cuda_flag)

# Train
learner.train(n_epochs=args.epochs, patience = args.patience)

# Recreate the model, load final state and test
tester = LearningLoop(model, optimizer, loss_funct,
                      test_loader=test_loader, cuda_flag=cuda_flag)

# Load final state
tester.load_state_file('./final_state.pt')

# Test
tester.test()

# Ploting loss vs iterations
plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_loss']))

#%% Plotting some weight
# A. Weights from Input layer to Hidden layer 1
w1 = tester.model.fc1.weight.data.cpu().numpy()
plt.figure()
for ix_w in range(25):
    tmp = np.reshape(w1[ix_w,:], [28,28])
    ax = plt.subplot(5,5, ix_w + 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(ix_w))
    plt.imshow(1- tmp, cmap='gray')    
    
# B. Weights from Hidden layer 1 to Hidden layer 2    
w2 = tester.model.fc2.weight.data.cpu().numpy()
plt.figure()
for ix_w in range(10):
    tmp = np.reshape(w2[ix_w,:], [5,5])
    ax = plt.subplot(2,5, ix_w + 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(ix_w))
    plt.imshow(1- tmp, cmap='gray')   
    
# C. Weights from Hidden layer 2 to Output layer
w3 = tester.model.fc3.weight.data.cpu().numpy()
plt.figure()
for ix_w in range(10):
    tmp = np.reshape(w3[ix_w,:], [1,10])
    ax = plt.subplot(10,1, ix_w + 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(ix_w))
    plt.imshow(1- tmp, cmap='gray')   
