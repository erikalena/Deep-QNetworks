import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

##################################
# Deep Q-Network
##################################


class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=4, input_size=84):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.input_size = self.compute_conv_output_size(input_size)
        self.fc = nn.Linear(64*self.input_size*self.input_size, 256)
        self.out = nn.Linear(256, num_actions)

    def forward(self, states):
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.out(x)

    def compute_conv_output_size(self, input_size):
        for conv in [self.conv1, self.conv2]:
            input_size = int( (input_size - conv.kernel_size[0] ) / conv.stride[0] + 1)

        return input_size
    
    def init_params(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, 0.1, 0.2)
    
    def device(self):
        return next(self.parameters()).device



##################################
# Environment
##################################

class SnakeEnv():

    def __init__(self, Lx, Ly, start = None, end = None):
        
        # World shape
        self.Ly, self.Lx = Lx, Ly
 
        # start and end positions
        self.start = [0,0] if start is None else start
        self.end = None if end is None else end
        self.current_state = self.start
        
        self.done = False

        # space of actions  [Down,  Up,  Right,Left] 
        self.actions = np.array([[1,0],[-1,0],[0,1],[0,-1]])
        self.num_actions = len(self.actions)
        self.body = []
        self.points = 1
        
    def reset(self):
        """
        Restart snake by setting current state to start
        """
        self.current_state = self.start
        self.body = []
        self.done = False
        self.points = 1
        
    def single_step(self, state, action):
        """
        Evolves the environment given action A and current state.
        """
        a = self.actions[action] # action is an integer in [0,1,2,3]
        S_new = copy.deepcopy(state)
        S_new[:2] += a

        # add a penalty for moving
        reward = -1

        # if the snake eats itself, add penalty 
        if S_new[:2] in self.body:
            self.done = True
            reward = -1000

        # update all the body segments in reverse order
        for i in range(len(self.body)-1,0,-1):
            self.body[i] = self.body[i-1]
        
        # update the first segment
        if len(self.body) > 0:
            self.body[0] = self.current_state

        # If we go out of the world, we enter from the other side
        if (S_new[0] == self.Ly):
            S_new[0] = 0
        elif (S_new[0] == -1):
            S_new[0] = self.Ly - 1
        elif (S_new[1] == self.Lx):
            S_new[1] = 0
        elif (S_new[1] == -1):
            S_new[1] = self.Lx - 1

        elif np.all(S_new[:2] == S_new[2:]):
            self.done = True       
            reward = 100  # if we reach the reward we get a reward of 100
            # add an element to the body
            self.points += 1 # if we eat the food we have a level of 1
            new_segment = self.body[-1] if len(self.body) > 0 else S_new[:2]
            self.body.append(new_segment)
        
        # change the current position
        self.current_state = S_new[:2]
        return S_new, reward, self.done
    
    def get_points(self):
        return self.points


    def get_image(self,state):
        """
        Represent the game as an image, state input is a tuple of 4 elements
        (x,y,x_food,y_food)
        """
        image = np.zeros((self.Lx,self.Ly))
        if state[2] >= 0 and state[2] < self.Lx and state[3] >= 0 and state[3] < self.Ly:
            image[int(state[2]), int(state[3])] = 1

        if state[0] >= 0 and state[0] < self.Lx and state[1] >= 0 and state[1] < self.Ly:
            image[int(state[0]), int(state[1])] = 1
        else:
            # if the agent is out of the world, it is dead and so we cancel the food as well
            # this check is just for safety reasons, if we allow the snake to go through the walls
            # this should never happen
            image[int(state[2]), int(state[3])] = 0 
            
        for i in range(len(self.body)):
            if self.body[i][0] >= 0 and self.body[i][0] < self.Lx and self.body[i][1] >= 0 and self.body[i][1] < self.Ly:
                image[int(self.body[i][0]), int(self.body[i][1])] = 1
            
        return image

    def select_epsilon_greedy_action(self, model, state, epsilon):
        """
        Take random action with probability epsilon, 
        else take best action.
        """
        result = np.random.uniform()
        if result < epsilon:
            return np.random.choice(np.arange(self.num_actions)) 
        else:
            # input is a tensor of floats
            images = self.get_image(state[0]) 
            logging.debug(f"model.device: {model.device()}")
            input = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(model.device())

            qs = model(input).cpu().data.numpy()
            return np.argmax(qs) 




   