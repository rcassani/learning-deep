#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is CNN classifier trained for the Dogs vs Cats InclassKaggle challenge 
for image classification. The challege was part of the assignment #1 for the 
course IFT 6135 Representation Learning Winter 2019
https://sites.google.com/mila.quebec/ift6135.
https://www.kaggle.com/c/ift6135h19/data
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
from PIL import Image
import os

import model_zoo                         # model to train 

# import scripts from the 'utils' directory 
import sys
sys.path.append('../../utils/')
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

# transformations
set_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

 
not_transforms = transforms.Compose([
    transforms.ToTensor(),
])

  
# traiset folder has 9999 samples for each class, i.e., 19998
dog_cat_dataset_mod = datasets.ImageFolder('../../data/dogs_vs_cats/trainset', transform=set_transforms)
dog_cat_dataset = datasets.ImageFolder('../../data/dogs_vs_cats/trainset', transform=not_transforms)


# splitting the dataset in train, validation 
dataset_size = len(dog_cat_dataset)
indices = list(range(dataset_size))
train_size = int(np.floor(0.7 * dataset_size))      # 70%  13998
valid_size = int(np.floor(0.2 * dataset_size))      # 20%   3999
test_size  = dataset_size - train_size - valid_size # 10%   2001

np.random.shuffle(indices)  # this does the shuffle in the data

# original dataset
train_dataset = data.Subset(dog_cat_dataset, indices[: train_size])
valid_dataset = data.Subset(dog_cat_dataset, indices[train_size : train_size + valid_size])
test_dataset  = data.Subset(dog_cat_dataset, indices[train_size + valid_size : ])

# transformed datasets
train_dataset_mod = data.Subset(dog_cat_dataset_mod, indices[: train_size])
valid_dataset_mod = data.Subset(dog_cat_dataset_mod, indices[train_size : train_size + valid_size])


# concatenate train and valid datasets
train_dataset = data.ConcatDataset([train_dataset, train_dataset_mod])
valid_dataset = data.ConcatDataset([valid_dataset, valid_dataset_mod])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.n_workers)
valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers)
test_loader  = DataLoader(test_dataset , batch_size=args.test_batch_size , shuffle=False, num_workers=args.n_workers)


#
#
#
#train_indices = indices[: train_size]
#valid_indices = indices[train_size : train_size + valid_size]
#test_indices  = indices[train_size + valid_size : ]
#
## data samplers
#train_sampler = data.SubsetRandomSampler(train_indices)
#valid_sampler = data.SubsetRandomSampler(valid_indices)
#test_sampler  = data.SubsetRandomSampler(test_indices)
#
## data loaders
#train_loader = DataLoader(dog_cat_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.n_workers)
#valid_loader = DataLoader(dog_cat_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, num_workers=args.n_workers)
#test_loader  = DataLoader(dog_cat_dataset, sampler=test_sampler, batch_size=args.test_batch_size , num_workers=args.n_workers)

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

#%% Ploting loss vs iterations
plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_loss']))
plt.plot(ix, np.array(tester.history['valid_loss']))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training','Validation'])

plt.figure()
ix = np.arange(tester.ix_epoch)
plt.plot(ix, np.array(tester.history['train_acc']))
plt.plot(ix, np.array(tester.history['valid_acc']))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training','Validation'])

#%% Plots 

# True and Predicted labels for some examples
examples = np.random.randint(test_size, size=40)
classes = dog_cat_dataset.classes
plt.figure()
for ix_example in range(len(examples)):
  index = examples[ix_example]
  X = test_dataset[index][0].numpy()
  X = np.transpose(X, axes=[1,2,0])
  y = test_dataset[index][1]
  ax = plt.subplot(5,8, ix_example + 1)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.title('T'+ classes[y] + ', P' + classes[y_hat[index]])
  plt.imshow(X)
#  

#%% get random image
image = Image.open(r'./poupoune.jpg'); ty = 0  #0cat 1dog

x = transforms.functional.to_tensor(image)
x.unsqueeze_(0)
if cuda_flag:
  x = x.cuda()
y = torch.tensor([[ty]])
print(x.shape)

output = tester.model.forward(x)
val, output = torch.max(output, dim=1)
out = output.item()


plt.figure()
plt.title(('predicted: ' + classes[out] + ',   {:.4f}').format(val.item()))
plt.imshow(image)
    