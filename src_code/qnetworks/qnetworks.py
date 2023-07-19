import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
from qlearning import *
from torch.utils.data import DataLoader
from collections import deque


class QNetwork(nn.Module):
    """
    Define a simple pytorch neural network, which takes as input 
    the state of the game (position in which we are i,j and the index of the reward) 
    and outputs the action to take (up, down, left, right)
    """
        
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
        

##################################
# Qlearning with experience replay
##################################

class QLearnNN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(QLearnNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_actions)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.out(x)





class ReplayBuffer(object):
    """
    Replay buffer to store past experiences that the 
    agent can then use for training the neural network.
    """

    def __init__(self, size, device:str = 'cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, state, body, action, reward, next_state, next_body, done):
        self.buffer.append((state, body, action, reward, next_state, next_body, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, bodies, actions, rewards, next_states, next_bodies, dones = [], [], [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
       
        idx[0] = np.random.randint(0, len(self.buffer)-(num_samples+1))
        idx[1:] = np.arange(idx[0]+1, idx[0]+num_samples)

        for i in idx:
            elem = self.buffer[i]
            state, body, action, reward, next_state, next_body, done = elem
            states.append(state)
            bodies.append(body)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_bodies.append(next_body)
            dones.append(done)

        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return states, bodies, actions, rewards, next_states, next_bodies, dones
    


##################################
# Custom dataset to use with dataloader
##################################

  
class customDataset(torch.utils.data.Dataset): #type: ignore
    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
  

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        
        return self.input[idx], self.labels[idx]
        