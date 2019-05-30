#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:20:44 2019

@author: cassani
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

import torch
import torch.nn as nn

import random
import time

import numpy as np




all_letters = string.ascii_letters + " .,;'-"
#n_letters = len(all_letters) + 1 # Plus EOS marker


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]



######
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        #self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        #output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
class vRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(vRNN, self).__init__()
        self.hidden_size = hidden_size

        self.u  = nn.Linear(input_size, hidden_size, bias = False)
        self.w  = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        a = self.w(hidden) + self.u(input)
        hidden = torch.tanh(a)
        output = self.v(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    

#########
# Random item from the list 'lines'
def get_random_sequence(sequences):
    return sequences[random.randint(0, len(sequences) - 1)]

def sequence_to_one_hot(sequence, char_to_ix):
    # number of features
    n_features = len(char_to_ix) 
    # encode sequence according dictionary
    sequence_encoded = encode_sequence(sequence, char_to_ix)
    # length sequence
    len_sequence = len(sequence_encoded)
    # Initialize the the encoded array
    one_hot = np.zeros((len_sequence, n_features)) 
    for ix, char_ix in enumerate(sequence_encoded):
        one_hot[ix, char_ix] = 1.0        
    # add dimension
    one_hot = np.expand_dims(one_hot, axis=1)     
    return one_hot
    
def encode_sequence(sequence, char_to_ix):
    sequence_encoded = np.array([char_to_ix[char] for char in sequence])
    return sequence_encoded

def get_target_sequence(sequence, char_to_ix):
    # number_features
    n_features = len(char_to_ix) 
    # encode sequence according dictionary
    sequence_encoded = encode_sequence(sequence, char_to_ix)
    sequence_encoded = np.concatenate((sequence_encoded[1:], np.array([n_features - 1])))
    return sequence_encoded
    
def get_training_pair(sequences, char_to_ix):
    sequence = get_random_sequence(sequences)
    sequence_in = sequence_to_one_hot(sequence, char_to_ix)
    sequence_out = get_target_sequence(sequence, char_to_ix)   
    # to tensors
    x = torch.FloatTensor(sequence_in)
    y = torch.LongTensor(sequence_out)
    return x, y

#########
def train(model, cuda_flag, x, y):   
    criterion = nn.NLLLoss()
    learning_rate = 0.005

    hidden = model.initHidden()

    if cuda_flag:
        x = x.cuda()
        y = y.cuda()
        hidden = hidden.cuda()

    y.unsqueeze_(-1)
    model.zero_grad()

    loss = 0
    for i in range(x.size(0)):
        output, hidden = model.forward(x[i], hidden)
        l = criterion(output, y[i])
        loss += l

    loss.backward()

    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / x.size(0)    



def timeSince(since):
    now = time.time()
    s = now - since
    m = np.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Sample from a category and starting letter
def sample(model, cuda_flag, start_letter, char_to_ix, ix_to_char):
    with torch.no_grad():  # no need to track history in sampling
        x = torch.FloatTensor(sequence_to_one_hot(start_letter, char_to_ix))
        if cuda_flag:
            x = x.cuda()
            
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = model.forward(x[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            topi = topi.item()
            if topi == len(char_to_ix) - 1: #EOS
                break
            else:
                letter = ix_to_char[topi]
                output_name += letter
            x = torch.FloatTensor(sequence_to_one_hot(letter, char_to_ix))

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(model, cuda_flag, start_letters, char_to_ix, ix_to_char):
    for start_letter in start_letters:
        print(sample(model, cuda_flag, start_letter, char_to_ix, ix_to_char))

######################
######################

# read Russian names
lines = readLines('../../data/names/Spanish.txt')

# generate vocabulary
voc_list = list(string.ascii_letters + " .,;'-") + ['*']
voc_size = len(voc_list)

# two dictionaries
ix_to_char = {ix:char for ix, char in enumerate(voc_list)}
char_to_ix = {char:ix for ix, char in enumerate(voc_list)}

# CUDA available
cuda_flag = torch.cuda.is_available()
cuda_flag = False

# RNN architecture
model = vRNN(voc_size, 128, voc_size)

if cuda_flag:
    model = model.cuda()


# Training parameters
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    x, y = get_training_pair(lines, char_to_ix)
    output, loss = train(model, cuda_flag, x, y)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

max_length = 20
samples(model, cuda_flag, 'ABCDEFGHI', char_to_ix, ix_to_char)














