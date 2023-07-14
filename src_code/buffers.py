import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
from src_code.qlearning import *
from torch.utils.data import DataLoader
from collections import deque



class SeqReplayBuffer(object):
    """
    Replay buffer to store past experiences that the 
    agent can then use for training the neural network.
    """

    def __init__(self, size, device:str = 'cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones= [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
       
        idx[0] = np.random.randint(0, len(self.buffer)-(num_samples+1))
        idx[1:] = np.arange(idx[0]+1, idx[0]+num_samples)

        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
            # args.append(rest)

        states = torch.as_tensor(np.array(states, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.array(actions,  dtype=np.float32), device=self.device)
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32), device=self.device)
        next_states = torch.as_tensor(np.array(next_states, dtype=np.float32), device=self.device)
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)
        # args = [torch.as_tensor(np.array(arg, dtype=np.float32), device=self.device) for arg in args]
        # args = tuple(args)
        return states, actions, rewards, next_states, dones #, args[0], args[1]


# class SeqReplayBuffer(ReplayBuffer):
#         def add(self, state, action, reward, next_state, done):
#             super().add(state, action, reward, next_state, done)

#         def sample(self, num_samples):
#             states, actions, rewards, next_states, dones, *args = super().sample(num_samples)
#             return states, actions, rewards, next_states, dones
        
        

class VecReplayBuffer(object): #TODO make it as a subclass of ReplayBuffer
    """
    Replay buffer to store past experiences that the 
    agent can then use for training the neural network.
    """

    def __init__(self, size, device:str = 'cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, state, action, reward, next_state, done, body, new_body):
        self.buffer.append((state, action, reward, next_state, done, body, new_body))
    
    def add_multiple(self, states, actions, rewards, next_states, dones, bodies, new_bodies):
        for i in range(len(states)):
            self.buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i], bodies[i], new_bodies[i]))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones, bodies, new_bodies = [], [], [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
       
        idx[0] = np.random.randint(0, len(self.buffer)-(num_samples+1))
        idx[1:] = np.arange(idx[0]+1, idx[0]+num_samples)

        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done, body, new_body = elem
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            bodies.append(body)
            dones.append(done)
            new_bodies.append(new_body)

        states = torch.as_tensor(np.array(states, dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.array(actions,  dtype=np.float32), device=self.device)
        rewards = torch.as_tensor(np.array(rewards, dtype=np.float32), device=self.device)
        next_states = torch.as_tensor(np.array(next_states, dtype=np.float32), device=self.device)
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)
        #bodies = torch.as_tensor(np.asarray(bodies,dtype=np.float32) , device=self.device)
        return states, actions, rewards, next_states, dones, bodies, new_bodies
