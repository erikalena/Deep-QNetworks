import torch
import numpy as np
from collections import deque
        
class SeqReplayBuffer(object):
    """
    Replay buffer to store past experiences that the 
    agent can then use for training the neural network.
    """

    def __init__(self, size, device:str = 'cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, state, action, reward, next_state, done, info, new_info):
        state = list(np.concatenate([state["agent"],state["target"]]))
        next_state = list(np.concatenate([next_state["agent"],next_state["target"]]))
        body = info["body"]
        new_body = new_info["body"]

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

