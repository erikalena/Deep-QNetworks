import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import gymnasium as gym
from gymnasium import spaces
import pygame
import os



##################################
# Deep Q-Network
##################################
class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=4, input_size=84):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
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




#####################
# Agent
#####################
class SnakeAgent:
    def __init__(
        self,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        num_actions: int,
        env: gym.Env,
        size: tuple[int, int],
        device,
        discount_factor: float = 0.95,
        num_envs: int = 1,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.num_actions = num_actions
        self.size = size

        self.env = env
        self.num_envs = num_envs
        
        self.device = device
        self.model = DQN(in_channels =1, num_actions=self.num_actions, input_size=self.size[0]).to(self.device)
        self.model_target = DQN(in_channels = 1, num_actions=self.num_actions, input_size=self.size[0]).to(self.device)


 
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.size = size
        
        self.env = env
        self.training_error = []


    def get_image(self, state, body):
        """
        Represent the game as an image, state input is a tuple of 4 elements
        (x,y,x_food,y_food)
        """
        image = np.zeros((self.size[0],self.size[1]))
        if state[2] >= 0 and state[2] < self.size[0] and state[3] >= 0 and state[3] < self.size[1]:
            image[int(state[2]), int(state[3])] = .5

        if state[0] >= 0 and state[0] < self.size[0] and state[1] >= 0 and state[1] < self.size[1]:
            image[int(state[0]), int(state[1])] += 1
        else:
            # if the agent is out of the world, it is dead and so we cancel the food as well
            # this check is just for safety reasons, if we allow the snake to go through the walls
            # this should never happen
            image[int(state[2]), int(state[3])] = 0 
            
        for i in range(len(body)):
            if body[i][0] >= 0 and body[i][0] < self.size[0] and body[i][1] >= 0 and body[i][1] < self.size[1]:
                image[int(body[i][0]), int(body[i][1])] += .1
        return image

        

    def get_action(self, state, info) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        state = list(np.concatenate([state["agent"],state["target"]]))
        body = info["body"]
        if np.random.random() < self.epsilon:
            a = self.env.action_space.sample()
            condition = True
            while condition:
                b = self.env.action_space.sample()
                if a != b:
                    condition = False
            return [a,b]

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            # input is a tensor of floats
            images = self.get_image(state, body) 
            input = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            qs = self.model(input).cpu().data.numpy()
            a = np.argpartition(qs[0], -2)[-2:]
            #print(f'qs: {qs}, a: {a}')
            
            # they might not be sorted
            if qs[0][a[0]]<qs[0][a[1]]:
                a = [a[1],a[0]]
        
            return a
    
    def get_mult_action(self, states, bodies) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        ret = []
        for state, body in zip(states, bodies):
            result = np.random.uniform()
            if result < self.epsilon:
                ret.append(self.env.single_action_space.sample()) 
            else:
                # input is a tensor of floats
                images = self.get_image(state, body) 
                input = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                qs = self.model(input).cpu().data.numpy()
                ret.append(np.argmax(qs))
        return ret
            


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def load_model(self, checkpoint_path, optimizer):
        logging.info("Loading model from checkpoint: {}".format(checkpoint_path))
        try:
            try:
                os.path.isfile(checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model_target.load_state_dict(checkpoint["model_state_dict"])
                logging.info("Model loaded")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info("Optimizer loaded")
            except:
                logging.warning("Optimizer not found, loading default optimizer")
                os.path.isfile(checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint)
                self.model_target.load_state_dict(checkpoint)
                logging.info("Model loaded")

        except:
            logging.warning("Checkpoint not found, loading default model")
        return optimizer
        
    def save_model(self, checkpoint_path, optimizer):
        """
        Save the model and optimizer parameters to disk.

        Args:
            checkpoint_path (str): The path to save the checkpoint file.
            optimizer (torch.optim.Optimizer): The optimizer used to train the model.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info("Model saved")
        