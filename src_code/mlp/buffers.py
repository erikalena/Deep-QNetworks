import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
from torch.utils.data import DataLoader
from collections import deque



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



