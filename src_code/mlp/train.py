import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy
import time
from buffers import ReplayBuffer
from agent import QLearnNN, train_step, select_epsilon_greedy_action
from environment import SnakeEnv




def main_loop(env, config):
    
    # initialize the Q-learning network
    model = QLearnNN(config['num_features'], config['num_actions']).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    # Start training. Play game once and then train with a batch.
    last_100_ep_rewards = []

    n_steps = np.zeros(config['num_episodes'])
    last_100_ep_rewards = []
    last_100_ep_steps = []
    food_eaten = []
    cur_frame = 0
    start_time = time.time()
    

    # load model from file
    #model.load_state_dict(torch.load("../../results/mlp/model.pth"))
    #optimizer.load_state_dict(torch.load("../../results/mlp/optimizer.pth"))

    step = 0
    epsilon = config['epsilon']

    for episode in range(config['num_episodes']):
        env.reset()
        # state is a random tuple of 4 values
        # made of starting position and goal position
        start_x = np.random.randint(0,config['Lx'])
        start_y = np.random.randint(0,config['Ly'])
        
        goal_x = np.random.randint(0,config['Lx'])
        goal_y = np.random.randint(0,config['Ly'])
    
        state = [start_x, start_y, goal_x, goal_y]
        body = [] # initialize body

        # running statistics
        ep_reward = 0
        done = False
        eaten = 0
        
        
        while not done and n_steps[episode] < max_steps_per_episode:
            
            action = select_epsilon_greedy_action(model, state, body, epsilon, config)
            
            next_state, new_body, reward, done = env.step(state, action, body)
            ep_reward += reward
            

            # Save to experience replay.
            buffer.add(state, body, action, reward, next_state, new_body, done)
            state = next_state
            cur_frame += 1
        
            # Train neural network.
            if len(buffer) > batch_size +1 and cur_frame % update_after_actions == 0:
                states, bodies, actions, rewards, next_states, new_bodies, dones = buffer.sample(batch_size)
                loss = train_step(model, config, states, actions, rewards, next_states, dones, bodies, new_bodies, discount, loss_fn, optimizer)
            
            if reward >= 1:
                eaten += 1
                goal_x = np.random.randint(0,config['Lx'])
                goal_y = np.random.randint(0,config['Ly'])
                state = [state[0], state[1], goal_x, goal_y] # if the snake eats the food, the goal changes

            if eaten == config['n_body']:
                done = True

            n_steps[episode] += 1
            step += 1
            if step % eps_decay_steps == 0:
                epsilon -= 0.01
                epsilon = max(epsilon, 0.1)

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
            last_100_ep_steps = last_100_ep_steps[1:]
            food_eaten = food_eaten[1:]
        last_100_ep_rewards.append(ep_reward)

        last_100_ep_steps.append(n_steps[episode])
        food_eaten.append(eaten)

        running_reward = np.mean(last_100_ep_rewards)
        mean_steps = np.mean(last_100_ep_steps)
        mean_food = np.mean(food_eaten)

        if episode % 200 == 0:
            # write on file
            time_taken = time.time() - start_time
            start_time = time.time()
            with open('../../results/mlp/results.txt', 'a') as f:
                f.write(f'{episode},{running_reward:.2f}, {epsilon:.3f}, {mean_steps:.3f}, {mean_food}, {time_taken:.3f}\n')

            # save the model
            # torch.save(model.state_dict(), '../../results/mlp/model_'+str(episode)+'.pth')        
        

        # at the end save the model along with the optimizer
        torch.save(model.state_dict(), '../../results/mlp/final_model.pth')
        torch.save(optimizer.state_dict(), '../../results/mlp/final_optimizer.pth')


if __name__ == "__main__":

    # snake environment parameters
    _Lx = 10
    _Ly = 10
    _n_body = 10  # maximum number of body segments
    _num_features = 4 + 2*_n_body # 4 for the head and goal, 2*n_body for the body
    _num_actions = 4
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Hyperparameters
    num_episodes = 50000
    max_steps_per_episode = 30000
    epsilon = 1.
    batch_size = 32
    discount = 0.99
    buffer = ReplayBuffer(100000, device=_device)
    cur_frame = 0
    
    update_after_actions = 2 # train the model every 2 steps
    eps_decay_steps = 100000 # decrease epsilon every 100000 steps

    # rewards
    positive_reward = 50
    negative_reward = -1

    # save all config in a json file
    config = {
        "Lx": _Lx,
        "Ly": _Ly,
        "n_body": _n_body,
        "num_features": _num_features,
        "num_actions": _num_actions,
        "device": _device,
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "epsilon": epsilon,
        "eps_decay_steps": eps_decay_steps,
        "batch_size": batch_size,
        "discount": discount,
        "update_after_actions": update_after_actions,
        "positive_reward": positive_reward,
        "negative_reward": negative_reward,
    }

    # write configuration to file
    f = open("../../results/mlp/results.txt", "w")
    for key in config.keys():
        f.write(key + ": " + str(config[key]) + "\n")

    f.write('episode,running_reward,epsilon,mean_steps,mean_food,time_taken\n') 
    f.close()
    
    env = SnakeEnv(_Lx,_Ly, _n_body, positive_reward=positive_reward, negative_reward=negative_reward)
    main_loop(env, config)
    

    
