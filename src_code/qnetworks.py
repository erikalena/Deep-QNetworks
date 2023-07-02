import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
from qlearning import *
from torch.utils.data import DataLoader

# construct a simple neural networks
# which takes as input the state of the game (position in which we are i,j and the index of the reward)
# and outputs the action to take (up, down, left, right)

# define a simple pytorch neural network
class QNetwork(nn.Module):
        
    def __init__(self, state_size, action_size, seed, fc1_units=16):
        
        # initialize the network
        super(QNetwork, self).__init__()
        
        # set the seed
        self.seed = torch.manual_seed(seed)
        
        # define the layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        # the output layer has as many units as the number of actions
        self.fc2 = nn.Linear(fc1_units, action_size)
        
        self.init_params()

    def forward(self, state):
        
        # define the forward pass
        x = F.relu(self.fc1(state))
        # add softmax to get probabilities
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
    def init_params(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, 0.1, 0.2)
        
        


class customDataset(torch.utils.data.Dataset):
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
  

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        
        return self.input[idx], self.labels[idx]
        