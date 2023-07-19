import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import deque



class SnakeEnv():
    def __init__(self, Lx, Ly, n_body, positive_reward=1, negative_reward=-1, start = None, end = None):
        
        # World shape
        self.Ly, self.Lx = Lx, Ly
 
        # policy is a matrix of size Ly x Lx x S 
        # where S is the number of states
        self.policy = np.zeros((self.Ly, self.Lx, self.Ly*self.Lx ))
        self.values = np.zeros((self.Ly, self.Lx, self.Ly*self.Lx ))

        # start and end positions
        self.start = [0,0] if start is None else start
        self.end = None if end is None else end

        # generate random positions for the reward
        self.final_pos = [np.random.randint(1, self.Ly), np.random.randint(1, self.Lx) ] if end is None else end
        
        # generate world instance
        reward = 10
        self.World = self.new_instance(Lx, Ly, self.final_pos, reward)
        
        # Keeps track of current state
        self.current_state = self.start
        
        # Keeps track of terminal state
        self.done = False

        #Actions = [Down,   Up,  Right,Left] 
        self.actions = np.array([[1,0],[-1,0],[0,1],[0,-1]])

        self.body = []
        self.n_body = n_body

        # rewards
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
    
    def new_instance(self, Lx, Ly, goal, rewards):
        World = np.zeros((Ly,Lx))
        World[goal[0], goal[1]] = rewards
        return World

        
    def reset(self):
        """
        Restart snake by setting current state to start
        """
        self.current_state = self.start

        self.body = []
        self.done = False
        
    def step(self, state, action, body):
        """
        Evolves the environment given action A and current state.
        """
        # Check if action A is in proper set
        a = self.actions[action] # action is an integer in [0,1,2,3]
        S_new = copy.deepcopy(state)
        head = S_new[:2]
        goal = S_new[2:4]
        head += a

        # add a penalty for moving
        reward = 0
        
        # if the snake eats itself, add penalty 
        # and stop the game
        if list(head) in body:
            #self.done = True
            reward = self.negative_reward

        # update all the body segments in reverse order
        # if body segment != -1, then update it
        for i in range(len(body)-1,0,-1):
            body[i] = list(body[i-1]).copy()

        # update the first segment
        if len(body) > 0:
            body[0] = list(state[:2]).copy()

        # If we go out of the world, we enter from the other side
        if (head[0] == self.Ly):
            head[0] = 0
        elif (head[0] == -1):
            head[0] = self.Ly - 1
        elif (head[1] == self.Lx):
            head[1] = 0
        elif (head[1] == -1):
            head[1] = self.Lx - 1
        
        elif np.all(head == goal):
            reward = self.positive_reward
            # add an element to the body
            if len(body) < self.n_body:
                new_segment = body[-1] if len(body) > 0 else list(head)
                body.append(new_segment)
        
        
        # change the current position
        self.current_state = head
        self.body = body
        S_new[:2] = head
        return S_new, body, reward, self.done











