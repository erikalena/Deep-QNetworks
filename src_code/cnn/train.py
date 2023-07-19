import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def train_step(states, actions, rewards, next_states, dones, bodies, new_bodies, snake_agent, loss_function, optimizer, device='cpu'):
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
    action_masks = F.one_hot(actions.long(), snake_agent.num_actions)
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_function(masked_qs, target.detach())
    snake_agent.training_error.append(loss.item())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss