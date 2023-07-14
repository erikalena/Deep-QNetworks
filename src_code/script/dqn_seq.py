import sys
sys.path.insert(0, '/home/alexserra98/uni/r_l/project/Deep-QNetworks/src_code')
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
from qnetworks import ReplayBuffer
from deep_qnetworks_vectorized import DQN , SnakeEnv, SnakeAgent
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32  # Size of batch taken from replay buffer
# initialize the environment 
Lx = 20
Ly = 20

env = SnakeEnv(size =(Lx,Ly))

# Initialize the models
model = DQN(in_channels =1, num_actions=env.action_space.n, input_size=env.Lx)
# The target model makes the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = DQN(in_channels = 1, num_actions=env.action_space.n, input_size=env.Lx)

model.to(device)
model_target.to(device)

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
# huber loss
loss_function = nn.HuberLoss()

# Train hyperparameters

cur_frame = 0
last_100_ep_rewards = []
max_steps_per_episode = 100
max_num_episodes = 10000

# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000

# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 100000.0

filename = 'dqn_results.txt'

# Initialize the agent
num_actions = env.action_space.n
#action_space = spaces.Discrete(4)
learning_rate = 0.01
initial_epsilon = 1.0

epsilon_decay = initial_epsilon / (max_num_episodes / 2) 
final_epsilon = 0.1
snake_agent = SnakeAgent(learning_rate, initial_epsilon, epsilon_decay, final_epsilon, env=env)


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
    images = [snake_agent.get_image(next_state, new_body) for next_state, new_body in zip(next_states,new_bodies)]
    input = torch.as_tensor(np.array(images), dtype=torch.float32).unsqueeze(1).to(device)
    max_next_qs = snake_agent.model_target(input).max(-1).values

    # if the next state is terminal, then the Q-value is just the reward
    # otherwise, we add the discounted max Q-value of the next state
    target = rewards + (1.0 - dones) * snake_agent.discount_factor * max_next_qs

    # then to compute the loss, we also need the Q-value of the current state
    images = [snake_agent.get_image(state, body) for state, body in zip(states,bodies)]
    input = torch.as_tensor(np.array(images), dtype=torch.float32).unsqueeze(1).to(device)
    qs = snake_agent.model(input)

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

for episode in tqdm(range(max_num_episodes)):
    env.reset()
    episode_reward = 0

    timestep = 0

    while timestep < max_steps_per_episode:
    
        cur_frame += 1
 
        observation = env._get_obs()
        info = env._get_info()
        body = info["body"]
        state = list(np.concatenate([observation["agent"],observation["target"]]))

        action = snake_agent.get_action(state, body)

        next_state, reward, done, new_body,_ = env.step(action)
        new_body = new_body["body"]
        next_state = list(np.concatenate([next_state["agent"],next_state["target"]]))

        episode_reward += reward
                
        # Save actions and states in replay buffer
        buffer.add(state, action, reward, next_state, done, body, new_body)

        cur_frame += 1

        # Train neural network.
        if len(buffer) > batch_size and cur_frame % update_after_actions == 0:
            states, actions, rewards, next_states, dones, bodies, new_bodies = buffer.sample(batch_size)
            loss = train_step(states, actions, rewards, next_states, dones, bodies, new_bodies, discount=0.99)
        
        # Update target network every update_target_network steps.
        if cur_frame % update_target_network == 0:
            snake_agent.model_target.load_state_dict(model.state_dict())

        timestep += 1
    
    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(episode_reward)

    running_reward = np.mean(last_100_ep_rewards)

    if episode+1 % 100 == 0:
        """ print(f'Episode {episode}/{max_num_episodes}. Epsilon: {epsilon:.3f}.'
        f' Reward in last 100 episodes: {running_reward:.2f}') """

        # write on file current average reward
        with open(filename, 'a') as f:
            f.write(f'{episode},{running_reward:.2f}, {snake_agent.epsilon:.3f}\n')

    # Condition to consider the task solved
    # e.g. to eat at least 6 consecutive food items
    # without eating itself, considering also the moves to reach the food
    if running_reward > 500: 
        print("Solved at episode {}!".format(episode))
        break