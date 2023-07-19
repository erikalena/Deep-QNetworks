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