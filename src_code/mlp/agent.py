import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




"""
Model for Q-learning with Neural Networks 
and experience replay.
"""
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
    

"""
Training function
"""


def train_step(model, config, states, actions, rewards, next_states, dones, bodies, new_bodies, discount, loss_fn, optimizer):
    """
    Perform a training iteration on a batch of data sampled from the experience
    replay buffer.
    """

    # compute targets for Q-learning
    # the max Q-value of the next state is the target for the current state
    # group next states and new bodies
    
    next_states = np.array(next_states)
    states = np.array(states)
    
    # not so straightforward to group states and bodies
    # we need to put them in a unique array, if the body is 
    # shorter than the maximum number of body segments, we need to pad it with zeros
    
    list_next_states = np.zeros((next_states.shape[0], config['num_features']))
    list_next_states[:,:next_states.shape[1]] = next_states + np.ones(next_states.shape)


    for j in range(next_states.shape[0]): # for each sample in batch
        # if new_bodies is not empty
        if len(new_bodies[j]) > 0:
            for i in range(len(new_bodies[j])): # for each body segment
                idx = 4 + i*2
                list_next_states[j][idx] = new_bodies[j][i][0] + 1.0
                list_next_states[j][idx+1] = new_bodies[j][i][1] + 1.0
    

    next_states = list_next_states        
    input = torch.as_tensor(next_states,  dtype=torch.float32)
    # normalize input
    input = input/(config['Lx']-1) + 0.001*torch.randn(input.shape)
    max_next_qs = model(input).max(-1).values

    # transform rewards to tensors
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    # transform dones to tensors
    dones = torch.as_tensor(dones, dtype=torch.float32)


    target = rewards + (1.0 - dones) * discount * max_next_qs

    # then to compute the loss, we also need the Q-value of the current state
    # group states and bodies
    list_states = np.zeros((states.shape[0], config['num_features']))
    list_states[:,:states.shape[1]] = states + np.ones(states.shape)
    for j in range(states.shape[0]):
        # if bodies is not empty
        if len(bodies[j]) > 0:
            for i in range(len(bodies[j])):
                idx = 4 + i*2
                list_states[j][idx] = bodies[j][i][0] + 1.0
                list_states[j][idx+1] = bodies[j][i][1] + 1.0
    
    states = list_states 
    input = torch.as_tensor(states, dtype=torch.float32)    
    input = input/(config['Lx']-1)  + 0.001*torch.randn(input.shape)

    qs = model(input)

    action_masks = F.one_hot(torch.as_tensor(actions).long(), config['num_actions'])
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_fn(masked_qs, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss



"""
Epsilon-greedy action selection.
"""

def select_epsilon_greedy_action(model, state, body, epsilon, config):
  """
  Take random action with probability epsilon, 
  else take best action.
  """
  result = np.random.uniform()
  if result < epsilon:
    return np.random.choice(np.arange(config['num_actions']))
  else:

    # read state and body and put them in a unique array
    for i in range(len(body)):
        state = np.append(state, body[i])
    
    if len(state) < config['num_features']:
        state = np.append(state+np.ones(len(state)), np.zeros((config['num_features'] - len(state))))

    # input is a tensor of floats
    input = torch.as_tensor(state, dtype=torch.float32)
    
    # normalize input
    input = input/(config['Lx'] -1)

    qs = model(input).cpu().data.numpy()
    return np.argmax(qs)
