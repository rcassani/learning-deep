#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is CNN classifier trained for the Dogs vs Cats InclassKaggle challenge 
for image classification. The challege was part of the assignment #1 for the 
course IFT 6135 Representation Learning Winter 2019
https://sites.google.com/mila.quebec/ift6135.
"""

import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import model_zoo                         # model to train 

# import scripts from the 'utils' directory 
import sys
sys.path.append('../../utils/')
from learning_loop import LearningLoop    # general learning loop

# Terminal: 
# python -i train_dnn.py --epochs=20 --ngpus=1 --lr=0.5 --l2=1 --model=small_cnn
# Spyder terminal
# runfile('train_dnn.py', '--epochs=20 --ngpus=1 --lr=0.5 --l2=1 --model=small_cnn')


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

# transformations
set_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# traiset folder has 9999 samples for each class
dog_cat_dataset = datasets.ImageFolder('../../data/dogs_vs_cats/trainset', transform=set_transforms)

# splitting the dataset in trai, validation and test
dataset_size = len(dog_cat_dataset)
indices = list(range(dataset_size))
valid_size = int(np.floor(0.2 * dataset_size))
test_size  = int(np.floor(0.1 * dataset_size))
train_size = dataset_size - valid_size - test_size 

np.random.shuffle(indices)
train_indices = indices[: train_size]
valid_indices = indices[train_size : train_size + valid_size]
test_indices  = indices[train_size + valid_size : ]

# Creating PT data samplers and loaders:
train_sampler = data.SubsetRandomSampler(train_indices)
valid_sampler = data.SubsetRandomSampler(valid_indices)
test_sampler  = data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(dog_cat_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.n_workers)
valid_loader = DataLoader(dog_cat_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, num_workers=args.n_workers)
test_loader  = DataLoader(dog_cat_dataset, sampler=test_sampler, batch_size=args.test_batch_size , num_workers=args.n_workers)

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

#%% Ploting loss vs iterations
plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_loss']))
plt.hold
plt.plot(ix, np.array(tester.history['valid_loss']))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training','Validation'])

plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_acc']))
plt.hold
plt.plot(ix, np.array(tester.history['valid_acc']))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training','Validation'])

#%% Plotting some weight
# A. Weights from Input layer to Hidden layer 1
kernels = np.squeeze(tester.model.features[0].weight.data.cpu().numpy())
plt.figure()
for ix_k in range(64):
    tmp = kernels[ix_k, :, :, :]
    tmp = np.moveaxis(tmp, [0,1,2], [2,0,1])
    ax = plt.subplot(8,8, ix_k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(ix_k))
    plt.imshow(1- tmp)    
    
#%% Feature map for few examples

for ix_e in range(5):
    plt.figure()
    
    # Plot random examples
    example_tuple = dog_cat_dataset.__getitem__(ix_e)    
    x_tensor = example_tuple[0].unsqueeze(0).cuda()
    x = np.squeeze(x_tensor.cpu().data.numpy())
    x = np.moveaxis(x, [0,1,2], [2,0,1])
    y = example_tuple[1]
    
    
    ax = plt.subplot(9,1, 9)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.title(str(y))
    plt.imshow(x) 
    
    tmp_tensor = tester.model.features[0](x_tensor)
    
    for ix_k in range(64):
        ax = plt.subplot(9,8, ix_k + 1)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.title(str(ix_k))        
        plt.imshow(tmp_tensor.data.cpu().numpy()[0, ix_k, :, :], cmap='gray') 
    
    

    