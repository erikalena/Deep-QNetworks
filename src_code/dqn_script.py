import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
from qnetworks import ReplayBuffer
from deep_qnetworks import DQN, SnakeEnv
import sys




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32  # Size of batch taken from replay buffer




def train_step(states, actions, rewards, next_states, dones, discount):
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
    images = [env.get_image(next_state) for next_state in next_states]
    input = torch.as_tensor(np.array(images), dtype=torch.float32).unsqueeze(1).to(device)
    max_next_qs = model_target(input).max(-1).values

    # if the next state is terminal, then the Q-value is just the reward
    # otherwise, we add the discounted max Q-value of the next state
    target = rewards + (1.0 - dones) * discount * max_next_qs

    # then to compute the loss, we also need the Q-value of the current state
    images = [env.get_image(state) for state in states]
    input = torch.as_tensor(np.array(images), dtype=torch.float32).unsqueeze(1).to(device)
    qs = model(input)

    # for each state, we update ONLY the Q-value of the action that was taken
    action_masks = F.one_hot(torch.as_tensor(np.array(actions)).long(), num_actions)
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_function(masked_qs, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss



def dqn_learning(env, filename):
    

    # initialize the buffer, with a size of 100000, when it is full, it will remove the oldest element
    buffer = ReplayBuffer(size = 100000, device=device) 

    cur_frame = 0
    last_100_ep_rewards = []
    max_steps_per_episode = 5000
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

    # print all configuration of file
    # open file 
    with open(filename, 'w') as f:
        # write all configuration
        f.write(f'Number of episodes: {max_num_episodes}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Max steps per episode: {max_steps_per_episode}\n')
        f.write(f'Update after actions: {update_after_actions}\n')
        f.write(f'Size of environment: {env.Lx}x{env.Ly}\n')
        f.write(f'Number of random frames: {epsilon_random_frames}\n')
        f.write(f'Number of greedy frames: {epsilon_greedy_frames}\n')
        f.write('episode, reward, epsilon\n')
    
    
    env.start = np.array([0,0])

    for episode in range(max_num_episodes):
        env.reset()
        episode_reward = 0

        # state is a tuple of 4 values made of starting position and goal position
        # start of an episode is always [0,0] for snake and a random position for goal
        start_x = env.start[0]
        start_y = env.start[1]
        goal_x = np.random.randint(0,env.Lx)
        goal_y = np.random.randint(0,env.Ly)
            
        state = [start_x, start_y, goal_x, goal_y]


        #done = False
        timestep = 0

        while timestep < max_steps_per_episode:
        
            cur_frame += 1

            state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
            action = env.select_epsilon_greedy_action(model, state_in, epsilon)
            
            next_state, reward, done = env.single_step(state, action)
            episode_reward += reward
        

            # Save actions and states in replay buffer
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            cur_frame += 1
        
            # Train neural network.
            if len(buffer) > batch_size and cur_frame % update_after_actions == 0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                loss = train_step(states, actions, rewards, next_states, dones, discount=0.99)
            
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

        if episode % 100 == 0:
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




if __name__ == "__main__":

    # read input arguments
    if len(sys.argv) > 1:
       Lx = Ly = int(sys.argv[1])

    else:
        Lx = Ly = 20

    # initialize the environment 
    env = SnakeEnv(Lx,Ly)

    model = DQN(in_channels =1, num_actions=env.num_actions, input_size=env.Lx)
    model_target = DQN(in_channels = 1, num_actions=env.num_actions, input_size=env.Lx)

    model.to(device)
    model_target.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    loss_function = nn.HuberLoss()

    num_actions = env.num_actions
    action_space = np.arange(num_actions)

    filename = 'dqn_results.txt'
    dqn_learning(env, filename)
