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


class SnakeEnv(gym.Env):

    metadata = {"render_modes": ["wb_array"], "render_fps": 4}
    
    def __init__(self, size, render_mode = "wb_array"):
        
        # World shape
        self.Ly, self.Lx = size
        #self.observation_space = spaces.Box(low=0, high=3, shape=[Lx, Ly])
        # start and end positions
        

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(np.array([0, 0]), np.array([self.Lx, self.Ly]), shape=(2,), dtype=int),
                "target": spaces.Box(np.array([0, 0]), np.array([self.Lx, self.Ly]), shape=(2,), dtype=int),
            }
        )
        
        self.start = [0,0]
        self.end = None
        #self.start = [0,0] if start is None else start
        #self.end = None if end is None else end # temporary, we don't want it to be unbounded
        #self._agent_location = self.start
        #self._target_location = self.start # temporary
        self.body = []

        self.done = False

        # space of actions  [Down,  Up,  Right,Left]
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([-1, 0]),
            2: np.array([0, 1]),
            3: np.array([0, -1]),
        }

        self.num_actions = 4
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"body": self.body}
        
    def reset(self,seed=None, options=None):
        """
        Restart snake by setting current state to start
        """
        super().reset(seed=seed)
        self._agent_location = [0,0]
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
            np.array([0, 0]), np.array([self.Lx, self.Ly]), size=(2,), dtype=int
        )

        self.body = []
        self.done = False
        observation = self._get_obs()
        info = self._get_info() 
        return observation, info
        
    def step(self, action):
        """
        Evolves the environment given action A and current state.
        """

        a = self._action_to_direction[action] # action is an integer in [0,1,2,3]


        self._agent_location += a

        # add a penalty for moving
        reward = -1

        # if the snake eats itself, add penalty 
        if self._agent_location in self.body:
            self.done = True
            reward = -1000

        # update all the body segments in reverse order
        for i in range(len(self.body)-1,0,-1):
            self.body[i] = self.body[i-1]
        
        # update the first segment
        if len(self.body) > 0:
            self.body[0] = self._agent_location

        # If we go out of the world, we enter from the other side
        if (self._agent_location[0] == self.Ly):
            self._agent_location[0] = 0
        elif (self._agent_location[0] == -1):
            self._agent_location[0] = self.Ly - 1
        elif (self._agent_location[1] == self.Lx):
            self._agent_location[1] = 0
        elif (self._agent_location[1] == -1):
            self._agent_location[1] = self.Lx - 1

        
        elif np.all(self._agent_location == self._target_location): ## this might not work
            self.done = True       
            reward = 100  # if we reach the reward we get a reward of 100
            print("Reached the target")
            # add an element to the body
            new_segment = self.body[-1] if len(self.body) > 0 else self._agent_location
            self.body.append(new_segment)
        
        # change the current position
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.done, info, False # I don't knwo what the last False is for, just overriding
    

    def render(self):
        if self.render_mode == "wb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """
        Represent the game as an image, state input is a tuple of 4 elements
        (x,y,x_food,y_food)
        """
        image = np.zeros((self.Lx,self.Ly))
        if self._target_location[0] >= 0 and self._target_location[0] < self.Lx and self._target_location[1] >= 0 and self._target_location[1] < self.Ly:
            image[int(self._target_location[0]), int(self._target_location[1])] = 1

        if self._agent_location[0] >= 0 and self._agent_location[0] < self.Lx and self._agent_location[1] >= 0 and self._agent_location[1] < self.Ly:
            image[int(self._agent_location[0]), int(self._agent_location[1])] = 1
        else:
            # if the agent is out of the world, it is dead and so we cancel the food as well
            # this check is just for safety reasons, if we allow the snake to go through the walls
            # this should never happen
            image[int(self._target_location[0]), int(self._target_location[1])] = 0 
            
        for i in range(len(self.body)):
            if self.body[i][0] >= 0 and self.body[i][0] < self.Lx and self.body[i][1] >= 0 and self.body[i][1] < self.Ly:
                image[int(self.body[i][0]), int(self.body[i][1])] = 1
            
        return image
    

#####################
# Agent
#####################
class SnakeAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        env: gym.Env,
        discount_factor: float = 0.95,
        size: tuple[int, int] = (20, 20),
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        self.device = device
        self.model = DQN(in_channels =1, num_actions=env.action_space.n, input_size=env.Lx).to(device)
        self.model_target = DQN(in_channels = 1, num_actions=env.action_space.n, input_size=env.Lx).to(device)


        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.size = size
        
        self.env = env

    def get_image(self, state, body):
        """
        Represent the game as an image, state input is a tuple of 4 elements
        (x,y,x_food,y_food)
        """
        image = np.zeros((self.size[0],self.size[1]))
        if state[2] >= 0 and state[2] < self.size[0] and state[3] >= 0 and state[3] < self.size[1]:
            image[int(state[2]), int(state[3])] = .5

        if state[0] >= 0 and state[0] < self.size[0] and state[1] >= 0 and state[1] < self.size[1]:
            image[int(state[0]), int(state[1])] = 1
        else:
            # if the agent is out of the world, it is dead and so we cancel the food as well
            # this check is just for safety reasons, if we allow the snake to go through the walls
            # this should never happen
            image[int(state[2]), int(state[3])] = 0 
            
        for i in range(len(body)):
            if body[i][0] >= 0 and body[i][0] < self.size[0] and body[i][1] >= 0 and body[i][1] < self.size[1]:
                image[int(body[i][0]), int(body[i][1])] = .1
        return image
            
        

    def get_action(self, state, body) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            # input is a tensor of floats
            images = self.get_image(state, body) 
            input = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.model.device)
            qs = self.model(input).cpu().data.numpy()
            return np.argmax(qs)
    
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
                ret.append(self.env.action_space.sample()) 
            else:
                # input is a tensor of floats
                images = self.get_image(state, body) 
                input = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(model.device)
                qs = self.model(input).cpu().data.numpy()
                ret.append(np.argmax(qs))
        return ret
            


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    