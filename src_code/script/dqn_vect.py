NUM_ENVS = 3
import sys
sys.path.insert(0, '/home/alexserra98/uni/r_l/project/Deep-QNetworks/src_code')
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
from qnetworks import ReplayBuffer
from deep_qnetworks_vectorized import DQN , SnakeEnv
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32  # Size of batch taken from replay buffer

##################################
# Initialize environment
##################################
Lx = 20
Ly = 20


envs =  gym.vector.AsyncVectorEnv([
    lambda: SnakeEnv(size =(Lx,Ly))
    for _ in range(NUM_ENVS)
], context='forkserver')


# The first model makes the predictions for Q-values which are used to make a action.
model = DQN(in_channels =1, num_actions=envs.single_action_space.n, input_size=envs.single_observation_space.spaces["agent"].high[0])
# The target model makes the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = DQN(in_channels = 1, num_actions=envs.single_action_space.n, input_size=envs.single_observation_space.spaces["agent"].high[0])

model.to(device)
model_target.to(device)

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
# huber loss
loss_function = nn.HuberLoss()

num_actions = envs.single_action_space.n
action_space = np.arange(num_actions)
envs.reset()

##################################
# Utility functions
##################################

def get_image(state,body):
    """
    Represent the game as an image, state input is a tuple of 4 elements
    (x,y,x_food,y_food)
    """
    image = np.zeros((Lx,Ly))
    if state[2] >= 0 and state[2] < Lx and state[3] >= 0 and state[3] < Ly:
        image[int(state[2]), int(state[3])] = 1

    if state[0] >= 0 and state[0] < Lx and state[1] >= 0 and state[1] < Ly:
        image[int(state[0]), int(state[1])] = 1
    else:
        # if the agent is out of the world, it is dead and so we cancel the food as well
        # this check is just for safety reasons, if we allow the snake to go through the walls
        # this should never happen
        image[int(state[2]), int(state[3])] = 0 
        
    for i in range(len(body)):
        if body[i][0] >= 0 and body[i][0] < Lx and body[i][1] >= 0 and body[i][1] < Ly:
            image[int(body[i][0]), int(body[i][1])] = 1
        
    return image

def select_epsilon_greedy_action(model, epsilon, states, bodies):
    """
    Take random action with probability epsilon, 
    else take best action.
    """
    ret = []
    for state, body in zip(states, bodies):
        result = np.random.uniform()
        if result < epsilon:
            ret.append(np.random.choice(np.arange(num_actions))) 
        else:
            # input is a tensor of floats
            images = get_image(state, body) 
            input = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(model.device)

            qs = model(input).cpu().data.numpy()
            ret.append(np.argmax(qs))
    return ret
##################################
# Training
##################################
def train_step(states, actions, rewards, next_states, dones, bodies, new_bodies, discount):
    """
    Perform a training iteration on a batch of data sampled from the experience
    replay buffer.

    Takes as input:
        - states: a batch of states
        - actions: a batch of actions
        - rewards: a batch of rewards
        - next_states: a batch of next states
        - dones: a batch of dones
        - discount: the discount factor, standard discount factor in RL to evaluate less long term rewards
    """

    # compute targets for Q-learning
    # the max Q-value of the next state is the target for the current state
    # the image to be fed to the network is a grey scale image of the world
    images = [get_image(next_state, new_body) for next_state, new_body in zip(next_states,new_bodies)]
    input = torch.as_tensor(np.array(images), dtype=torch.float32).unsqueeze(1).to(device)
    max_next_qs = model_target(input).max(-1).values

    # if the next state is terminal, then the Q-value is just the reward
    # otherwise, we add the discounted max Q-value of the next state
    target = rewards + (1.0 - dones) * discount * max_next_qs

    # then to compute the loss, we also need the Q-value of the current state
    images = [get_image(state, body) for state, body in zip(states,bodies)]
    input = torch.as_tensor(np.array(images), dtype=torch.float32).unsqueeze(1).to(device)
    qs = model(input)

    # for each state, we update ONLY the Q-value of the action that was taken

    #action_masks = F.one_hot(torch.as_tensor(np.array(actions)).long(), num_actions)
    action_masks = F.one_hot(actions.long(), num_actions)
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_function(masked_qs, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# initialize the buffer, with a size of 100000, when it is full, it will remove the oldest element
buffer = ReplayBuffer(size = 100000, device=device) 

cur_frame = 0
last_100_ep_rewards = []
max_steps_per_episode = 100
max_num_episodes = 10000

epsilon = 1.0
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter

# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000

# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 100000.0

filename = 'dqn_results.txt'

#env.start = np.array([0,0])

for episode in tqdm(range(max_num_episodes)):
    envs.reset()
    episode_reward = np.zeros(NUM_ENVS)

    timestep = 0

    while timestep < max_steps_per_episode:
    
        cur_frame += 1

        agent = np.stack(envs.get_attr("_agent_location"))
        target = np.stack(envs.get_attr("_target_location"))
        states = np.concatenate((agent,target),axis=1)
        bodies = list(envs.get_attr("body"))
        actions = select_epsilon_greedy_action(model, 1.0, states, bodies)
        
        
        
        next_states, rewards, dones, _, new_bodies = envs.step(actions)
        next_states = np.concatenate((next_states["agent"],next_states["target"]),axis=1)
        episode_reward += rewards
        new_bodies = list(new_bodies["body"])
        
        # Save actions and states in replay buffer
        buffer.add_multiple(states, actions, rewards, next_states, dones, bodies, new_bodies)

        cur_frame += 1

        # Train neural network.
        if len(buffer) > batch_size and cur_frame % update_after_actions == 0:
            states, actions, rewards, next_states, dones, bodies, new_bodies = buffer.sample(batch_size)
            loss = train_step(states, actions, rewards, next_states, dones, bodies, new_bodies, discount=0.99)
        
        # Update target network every update_target_network steps.
        if cur_frame % update_target_network == 0:
            model_target.load_state_dict(model.state_dict())

        timestep += 1

        if timestep > epsilon_random_frames:
            epsilon -= (epsilon_max - epsilon_min) / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
    

    
    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(episode_reward)

    running_reward = np.mean(last_100_ep_rewards)

    if episode+1 % 100 == 0:
        """ print(f'Episode {episode}/{max_num_episodes}. Epsilon: {epsilon:.3f}.'
        f' Reward in last 100 episodes: {running_reward:.2f}') """

        # write on file current average reward
        with open(filename, 'a') as f:
            f.write(f'{episode},{running_reward:.2f}, {epsilon:.3f}\n')

    # Condition to consider the task solved
    # e.g. to eat at least 6 consecutive food items
    # without eating itself, considering also the moves to reach the food
    if running_reward > 500: 
        print("Solved at episode {}!".format(episode))
        break