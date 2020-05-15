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
import os

import model_zoo                         # model to train 

# import scripts from the 'utils' directory 
import sys
sys.path.append('../../utils/')
from dataloaders_mnist_ram import Loader
from learning_loop import LearningLoop    # general learning loop

# Terminal: 
# python -i train_dnn.py --epochs=100 --ngpus=1 --lr=0.5 --l2=1 --model=small_cnn
# Spyder terminal
# runfile('train_dnn.py', '--epochs=100 --ngpus=1 --lr=0.5 --l2=1 --model=small_cnn')


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

parser.add_argument('--ckpt-path', type=str, default='./ckpts', metavar='PATH', help='Directory for checkpoint files, if None or empty string, checkpoints are not saved')
parser.add_argument('--ckpt-load', type=int, default=None, metavar='N', help='Indicates epoch to restart. If None, training starts from scratch')

parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--n-workers', type=int, default=1, metavar='N', help='Number of workers for dataloaders')

parser.add_argument('--final-state', action='store_true', help='If this flag is present and the final state file exists, it is loaded')

args = parser.parse_args()

# validate ckpt-path input
if args.ckpt_path == '' or args.ckpt_path == 'None':
  args.ckpt_path = None

# Verify CUDA
cuda_flag = args.ngpus > 0 and torch.cuda.is_available()
print('CUDA Mode is: ' +  str(cuda_flag))

# Random seed
torch.manual_seed(args.seed)
if cuda_flag:
  torch.cuda.manual_seed(args.seed)

# Datasets and DataLoaders
train_dataset = Loader('train')
valid_dataset = Loader('validation')
test_dataset  = Loader('test')

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.n_workers)
valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers)
test_loader  = DataLoader(test_dataset , batch_size=args.test_batch_size , shuffle=False, num_workers=args.n_workers)

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

if not (args.final_state and os.path.exists('./final_state_' + args.model + '.pt')):
  # 3. Create training loop and train
  learner = LearningLoop(args.model, model, optimizer, loss_funct,
                       train_loader=train_loader, valid_loader=valid_loader,
                       ckpt_path=args.ckpt_path, ckpt_load=args.ckpt_load,
                       cuda_flag=cuda_flag)

  # Train
  learner.train(n_epochs=args.epochs, patience = args.patience)

# Recreate the model, load final state and test
tester = LearningLoop(args.model, model, optimizer, loss_funct,
                      test_loader=test_loader, cuda_flag=cuda_flag)

# Load final state
tester.load_state_file( './final_state_' + args.model + '.pt')

# Test
y_hat_tensor = tester.test()
y_hat = y_hat_tensor.cpu().numpy()


#%% Plots
# Ploting loss vs iterations
plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_loss']))
plt.plot(ix, np.array(tester.history['valid_loss']))
plt.legend(['training set', 'validation set'])
plt.xlabel('epochs')
plt.ylabel('loss')

# Ploting acc vs iterations
plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_acc']))
plt.plot(ix, np.array(tester.history['valid_acc']))
plt.legend(['training set', 'validation set'])
plt.xlabel('epochs')
plt.ylabel('accuracy')


#%% Plotting some weight
# A. Weights
kernels = np.squeeze(tester.model.features[0].weight.data.cpu().numpy())
plt.figure()
for ix_k in range(10):
    tmp = kernels[ix_k, :, :]
    ax = plt.subplot(4,3, ix_k + 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(ix_k))
    plt.imshow(1- tmp, cmap='gray')    
    
#%% Feature map for few examples
for ix_e in range(5):
    plt.figure()
    
    # Plot random examples
    example_tuple = train_dataset.__getitem__(ix_e)    
    x_tensor = example_tuple[0].view([1,1,28,28])
    x = np.squeeze(x_tensor.data.numpy())
    y = example_tuple[1].data.numpy()
    
    
    ax = plt.subplot(4,3, 12)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(y))
    plt.imshow(x, cmap='gray') 
    
    tmp_tensor = tester.model.features[0](x_tensor.cuda()).cpu()
    
    for ix_k in range(10):
        ax = plt.subplot(4,3, ix_k + 1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(str(ix_k))
        plt.imshow(tmp_tensor.data.numpy()[0, ix_k, :, :], cmap='gray') 
    
# True and Predicted labels for some examples
examples = np.random.randint(10000, size=40)
plt.figure()
for ix_example in range(len(examples)):
  index = examples[ix_example]
  X = test_dataset.samples[index]
  y = test_dataset.labels[index] 
  tmp = np.reshape(X, [28,28])
  ax = plt.subplot(5,8, ix_example + 1)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.title('T'+ str(y) + ', P' + str(y_hat[index]))
  plt.imshow(tmp, cmap='gray')   

    