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

##################################
# Environment
##################################
REWARD = 10
NEGATVIE_REWARD = -1

class SnakeEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, size, config, render_mode = "human"):
        
        # World shape
        self.config = config
        self.Ly, self.Lx = size
        self.window_size = 512  # The size of the PyGame window
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(np.array([0, 0]), np.array([self.Lx, self.Ly]), shape=(2,), dtype=int), # type: ignore
                "target": spaces.Box(np.array([0, 0]), np.array([self.Lx, self.Ly]), shape=(2,), dtype=int), # type: ignore
            }
        )
        

        #self.start = [0,0] if start is None else start
        #self.end = None if end is None else end # temporary, we don't want it to be unbounded
        #self._agent_location = self.start
        #self._target_location = self.start # temporary
        self.body = []
        
        # If no_back is True, the agent cannot go back to the previous position
        self.no_back = config.no_back

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
        self.eaten_fruits = 0
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
   
       
        

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {"body": self.body, "eaten_fruits": self.eaten_fruits, "done": self.done}
    
    def set_obs(self, obs):
        self._agent_location = obs["agent"]
        self._target_location = obs["target"]
    
    def set_info(self, info):
        self.body = info["body"]
        self.eaten_fruits = info["eaten_fruits"]
        self.done = info["done"]
        
    def reset(self,seed=None, options=None):
        """
        Restart snake by setting current state to start
        """
        super().reset(seed=seed)
        self._agent_location = np.asarray([0,0])
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
            np.array([0, 0]), np.array([self.Lx, self.Ly]), size=(2,), dtype=int
        )

        self.body = []
        self.eaten_fruits = 0
        self.done = False
        observation = self._get_obs()
        info = self._get_info() 
        return observation, info
    
    def check_collision(self):
        finding_list = [np.array_equal(self._agent_location,x) for x in self.body]
        if True in finding_list:
            return True, finding_list.index(True)
        else:
            return False, -1

    def step(self, action):
        """
        Evolves the environment given action A and current state.
        """
        selected_action = action[0]
        a = self._action_to_direction[action[0]] # action is an integer in [0,1,2,3]
        self._prev_agent_location = copy.deepcopy(self._agent_location)
        
        # move the agent
        self._agent_location += a

        # add a penalty for moving
        reward = self.config.reward["step"]

        # Out of bounds case
        if (self._agent_location[0] == self.Ly):
            self._agent_location[0] = 0
        elif (self._agent_location[0] == -1):
            self._agent_location[0] = self.Ly - 1
        elif (self._agent_location[1] == self.Lx):
            self._agent_location[1] = 0
        elif (self._agent_location[1] == -1):
            self._agent_location[1] = self.Lx - 1
        
        # if self.no_back = True and the agent is trying to go back, we select the second action
        if self.no_back and len(self.body) > 0 and  np.all(self._agent_location == self.body[-1]):
            self._agent_location = copy.deepcopy(self._prev_agent_location)
            a = self._action_to_direction[action[1]] 
            self._agent_location += a
            # Out of bounds case
            if (self._agent_location[0] == self.Ly):
                self._agent_location[0] = 0
            elif (self._agent_location[0] == -1):
                self._agent_location[0] = self.Ly - 1
            elif (self._agent_location[1] == self.Lx):
                self._agent_location[1] = 0
            elif (self._agent_location[1] == -1):
                self._agent_location[1] = self.Lx - 1
            selected_action = action[1]

        
        # add a penalty for moving
        reward = 0 # -1
        
        # Target reached case
        if np.all(self._agent_location == self._target_location): ## this might not work
            #self.done = True       
            reward = self.config.reward["eat"]  # if we reach the reward we get a reward of 100
            self.eaten_fruits += 1
            # add an element to the body
            self.body.append(self._prev_agent_location)
            #update the target location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(
                np.array([0, 0]), np.array([self.Lx, self.Ly]), size=(2,), dtype=int)
        # Collision case
        elif (self.check_collision())[0]: # I used warlus operator to avoid building tmp twice
            self.done = self.config.done_on_collision
            self.body.append(self._prev_agent_location)
            self.body = self.body[1:]
            reward = self.config.reward["dead"]
        else:
            if len(self.body) > 0:
                self.body.append(self._prev_agent_location)
                # remove the last element of the body
                self.body = self.body[1:]

        
        # change the current position
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.done, selected_action, info # I don't knwo what the last False is for, just overriding
    


    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            print("Initializing pygame")
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
            self.window_size / self.Lx
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 102, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Now we draw the body
        for body_part in self.body:
            pygame.draw.circle(
                canvas,
                (0, 51, 0),
                (body_part + 0.5) * pix_square_size,
                pix_square_size / 3,
            )
        # Finally, add some gridlines
        for x in range(self.Lx + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def play(self):
        self._render_frame()
        game_over=False
        while not game_over:
            
            for event in pygame.event.get():
                
                if event.type==pygame.QUIT:
                    game_over=True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.step(3)
                    elif event.key == pygame.K_DOWN:
                        self.step(2)
                    elif event.key == pygame.K_LEFT:
                        self.step(1)
                    elif event.key == pygame.K_RIGHT:
                        self.step(0)
                self._render_frame()
        
        self.close()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None


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
        