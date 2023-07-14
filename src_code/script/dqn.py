from src_code.buffers import VecReplayBuffer, SeqReplayBuffer
from src_code.deep_qnetworks import (
    DQN,
    SnakeEnv,
    GymSnakeEnv,
    get_image,
    select_epsilon_greedy_action,
)
from src_code.utils import add_parser_argument, VecStatistics, SeqStatistics

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
import numpy as np

import os
import argparse
import json
import datetime


from tqdm import tqdm
import logging
from dataclasses import dataclass


@dataclass
class Config:
    current_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32  # Size of batch taken from replay buffer
    env_size_x: int = 20
    env_size_y: int = 20
    num_envs: int = 5
    max_steps_per_episode: int = 10000
    max_num_episodes: int = 1000
    epsilon: float = 1.0
    epsilon_min: float = 0.1  # Minimum epsilon greedy parameter
    epsilon_max: float = 1.0  # Maximum epsilon greedy parameter
    update_after_actions: int = 4  # Train the model after 4 actions
    update_target_network: int = 10000  # How often to update the target network
    epsilon_random_frames: int = (
        50000  # Number of frames to take random action and observe output
    )
    epsilon_greedy_frames: float = 100000.0  # Number of frames for exploration
    output_filename: str = "log.json"
    output_logdir: str = "results/prova"
    output_checkpoint_dir: str = "checkpoints/prova"
    save_step: int = 50  # Save model every 100 episodes and log results
    logging_level: int = logging.DEBUG
    description: str = "Deep Q-Network Snake, done after snake eats itself"


CONFIG = Config()


##################################
# Training
##################################
def compute_targets_and_loss(
    model,
    model_target,
    loss_function,
    states,
    actions,
    rewards,
    next_states,
    dones,
    discount,
    env=None,
    bodies=None,
    new_bodies=None,
):
    # compute targets for Q-learning
    # the max Q-value of the next state is the target for the current state
    # the image to be fed to the network is a grey scale image of the world
    if env is not None:
        images = [env.get_image(next_state) for next_state in next_states]
    else:
        images = [
            get_image(next_state, CONFIG)
            for next_state in next_states # type: ignore
        ]

    input = (
        torch.as_tensor(np.array(images), dtype=torch.float32)
        .unsqueeze(1)
        .to(CONFIG.device)
    )

    max_next_qs = model_target(input).max(-1).values

    # transform into tensors and move to device
    rewards = torch.as_tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.as_tensor(dones, dtype=torch.float32).to(device)

    # if the next state is terminal, then the Q-value is just the reward
    # otherwise, we add the discounted max Q-value of the next state
    target = rewards + (1.0 - dones) * discount * max_next_qs

    # then to compute the loss, we also need the Q-value of the current state
    if env is not None:
        images = [env.get_image(state) for state in states]
    else:
        images = [get_image(state, CONFIG) for state in states]  # type: ignore

    input = (
        torch.as_tensor(np.array(images), dtype=torch.float32)
        .unsqueeze(1)
        .to(CONFIG.device)
    )

    qs = model(input)

    # for each state, we update ONLY the Q-value of the action that was taken
    action_masks = F.one_hot(actions.long(), num_actions)
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_function(masked_qs, target.detach())

    return target, loss


def vec_train_step(
    states,
    actions,
    rewards,
    next_states,
    dones,
    bodies=None,
    new_bodies=None,
    discount=0.99,
):
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

    target, loss = compute_targets_and_loss(
        model,
        model_target,
        loss_function,
        states,
        actions,
        rewards,
        next_states,
        dones,
        discount,
        bodies=bodies,
        new_bodies=new_bodies,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def seq_train_step(env, states, actions, rewards, next_states, dones, discount=0.99):
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

    target, loss = compute_targets_and_loss(
        model,
        model_target,
        loss_function,
        states,
        actions,
        rewards,
        next_states,
        dones,
        discount,
        env=env,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def sequential_learning(env, buffer, filename):
    cur_frame = 0
    statistics = SeqStatistics()
    pbar = tqdm(range(CONFIG.max_num_episodes), desc="Run episodes")
    epsilon = CONFIG.epsilon

    for episode in range(CONFIG.max_num_episodes):
        env.reset()
        episode_reward = 0

        # state is a tuple of 4 values made of starting position and goal position
        # start of an episode is always [0,0] for snake and a random position for goal
        start_x = env.start[0]
        start_y = env.start[1]
        goal_x = np.random.randint(0, env.Lx)
        goal_y = np.random.randint(0, env.Ly)

        body = []
        state = [start_x, start_y, goal_x, goal_y, []]

        # done = False
        timestep = 0
        accumulated_loss = 0
        while timestep < CONFIG.max_steps_per_episode:
            cur_frame += 1

            #state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(CONFIG.device)
            action = env.select_epsilon_greedy_action(model, state, epsilon)

            next_state, reward, done = env.single_step(state, action)
            episode_reward += reward

            # Save actions and states in replay buffer
            buffer.add(state, action, reward, next_state, done)  # type: ignore
            state = next_state
            cur_frame += 1

            # Train neural network.
            if (
                len(buffer) > CONFIG.batch_size
                and cur_frame % CONFIG.update_after_actions == 0
            ):
                states, actions, rewards, next_states, dones = buffer.sample(  # type: ignore
                    CONFIG.batch_size
                )
                loss = seq_train_step(
                    env, states, actions, rewards, next_states, dones, discount=0.99
                )
                accumulated_loss += loss.item()
            # Update target network every update_target_network steps.
            if cur_frame % CONFIG.update_target_network == 0:
                model_target.load_state_dict(model.state_dict())

            timestep += 1

            if cur_frame > CONFIG.epsilon_random_frames:
                epsilon -= (
                    CONFIG.epsilon_max - CONFIG.epsilon_min
                ) / CONFIG.epsilon_greedy_frames
                epsilon = max(epsilon, CONFIG.epsilon_min)

            if done:
                break

        if statistics.len == CONFIG.save_step:
            statistics.shift()
        statistics.append(
            loss=accumulated_loss / timestep,
            points=env.get_points(),
            steps=timestep,
            steps_per_point=env.n_steps_per_point,
            episode_reward=episode_reward,
        )

        """ if episode % 100 == 0:
            epsilon -= 0.025
            epsilon = max(epsilon, CONFIG.epsilon_min) """

        if episode % CONFIG.save_step == 0:
            """print(f'Episode {episode}/{max_num_episodes}. Epsilon: {epsilon:.3f}.'
            f' Reward in last 100 episodes: {running_reward:.2f}')"""
            logging.info(f"\n Saving results at episode {episode}")
            file = json.load(open(filename))
            file["episode_{}".format(episode)] = {
                "epsilon": epsilon,
                "points": env.get_points(),
                "steps": timestep,
                "steps_per_point": env.n_steps_per_point,
                "reward_mean": np.mean(statistics.episode_rewards),
                "loss_mean": np.mean(statistics.losses),
                "points_mean": np.mean(statistics.points),
                "steps_mean": np.mean(statistics.steps),
                "estimated_time": pbar.format_dict["elapsed"],
            }
            json.dump(file, open(filename, "w"), indent=4)
            # Save model
            torch.save(
                model.state_dict(),
                f"{CONFIG.output_checkpoint_dir}/model_{episode}.pth",
            )

        # Condition to consider the task solved
        # e.g. to eat at least 6 consecutive food items
        # without eating itself, considering also the moves to reach the food
        if np.mean(statistics.episode_rewards) > 500:
            logging.info("Solved at episode {}!".format(episode))
            break
        pbar.update(1)


def vectorized_learning(env, buffer, filename):
    cur_frame = 0
    statistics = VecStatistics()

    with open(filename, "w") as f:
        dict_json = {"configuration": CONFIG.__dict__}
        json.dump(dict_json, f, indent=4)

    epsilon = CONFIG.epsilon

    # env.start = np.array([0,0])
    pbar = tqdm(range(CONFIG.max_num_episodes))
    for episode in range(CONFIG.max_num_episodes):
        env.reset()
        episode_reward = np.zeros(CONFIG.num_envs)

        timestep = 0
        accumulated_loss = 0
        while timestep < CONFIG.max_steps_per_episode:
            cur_frame += 1

            agent = np.stack(env.get_attr("_agent_location"))
            target = np.stack(env.get_attr("_target_location"))
            states = np.concatenate((agent, target), axis=1)
            bodies = list(env.get_attr("body"))
            actions = select_epsilon_greedy_action(
                model, 1.0, states, bodies, num_actions, CONFIG
            )

            next_states, rewards, dones, _, new_bodies = env.step(actions)
            next_states = np.concatenate(
                (next_states["agent"], next_states["target"]), axis=1
            )
            episode_reward += rewards
            new_bodies = list(new_bodies["body"])
            
            if np.all(dones):
                break

            # Save actions and states in replay buffer
            buffer.add_multiple(
                states, actions, rewards, next_states, dones, bodies, new_bodies
            )

            cur_frame += 1

            # Train neural network.
            if (
                len(buffer) > CONFIG.batch_size
                and cur_frame % CONFIG.update_after_actions == 0
            ):
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    bodies,
                    new_bodies,
                ) = buffer.sample(CONFIG.batch_size)
                loss = vec_train_step(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    bodies,
                    new_bodies,
                    discount=0.99,
                )
                accumulated_loss += loss.item()
            # Update target network every update_target_network steps.
            if cur_frame % CONFIG.update_target_network == 0:
                model_target.load_state_dict(model.state_dict())

            timestep += 1

            if timestep > CONFIG.epsilon_random_frames:
                epsilon -= (
                    CONFIG.epsilon_max - CONFIG.epsilon_min
                ) / CONFIG.epsilon_greedy_frames
                epsilon = max(epsilon, CONFIG.epsilon_min)

        if statistics.len == CONFIG.save_step:
            statistics.shift()
        statistics.append(
            loss=accumulated_loss / timestep, episode_reward=episode_reward
        )

        if episode + 1 % 100 == 0:
            """print(f'Episode {episode}/{max_num_episodes}. Epsilon: {epsilon:.3f}.'
            f' Reward in last 100 episodes: {running_reward:.2f}')"""
            logging.info(f"\n Saving results at episode {episode}")
            file = json.load(open(filename))
            file["episode_{}".format(episode)] = {
                "epsilon": epsilon,
                # "points": env.get_points(),
                "steps" : timestep,
                # "steps_per_point": env.n_steps_per_point,
                "reward_mean": np.mean(statistics.episode_rewards),
                "loss_mean": np.mean(statistics.losses),
                # "points_mean": running_points,
                # "steps_mean": running_steps,
                "estimated_time": pbar.format_dict["elapsed"],
            }
            json.dump(file, open(filename, "w"), indent=4)
            # Save model
            torch.save(
                model.state_dict(),
                f"{CONFIG.output_checkpoint_dir}/model_{episode}.pth",
            )

        # Condition to consider the task solved
        # e.g. to eat at least 6 consecutive food items
        # without eating itself, considering also the moves to reach the food
        if np.mean(statistics.episode_rewards) > 500:
            print("Solved at episode {}!".format(episode))
            break
        pbar.update(1)


def dqn_learning(env):
    filename = f"{CONFIG.output_logdir}/{CONFIG.output_filename}"
    with open(filename, "w") as f:
        dict_json = {"configuration": CONFIG.__dict__}
        json.dump(dict_json, f, indent=4)

    epsilon = CONFIG.epsilon
    # env.start = np.array([0, 0])

    if CONFIG.num_envs == 1:
        logging.debug(f"dqn : sequential learning")
        buffer = SeqReplayBuffer(size=100000, device=CONFIG.device)
        sequential_learning(env, buffer, filename)
    else:
        logging.debug(f"dqn : vectorized learning")
        buffer = VecReplayBuffer(size=100000, device=CONFIG.device)
        vectorized_learning(env, buffer, filename)


if __name__ == "__main__":
    # read input arguments
    parser = argparse.ArgumentParser(description="Deep Q-Network Snake")
    parser = add_parser_argument(parser, CONFIG)
    args = parser.parse_args()

    CONFIG = Config(
        current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        logging_level=logging.DEBUG if args.debug else CONFIG.logging_level,
        env_size_x=args.Lx,
        env_size_y=args.Ly,
        num_envs=args.num_envs,
        max_num_episodes=args.episodes,
        batch_size=args.batch_size,
        max_steps_per_episode=args.max_steps,
        update_after_actions=args.update_after,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        update_target_network=args.update_target,
        epsilon_random_frames=args.epsilon_random_frames,
        epsilon_greedy_frames=args.epsilon_greedy_frames,
        output_logdir=args.output_log_dir,
        output_checkpoint_dir=args.output_checkpoint_dir,
        description="Vectorized DQN:" + args.desc
        if args.num_envs > 1
        else "Sequential DQN:" + args.desc,
    )

    logging.basicConfig(level=CONFIG.logging_level)
    logging.info(f"Start training with configuration: {CONFIG}")
    # make directory for checkpoints and results
    os.makedirs(CONFIG.output_logdir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)

    # initialize the environment
    if CONFIG.num_envs == 1:  # sequential
        env = SnakeEnv(CONFIG.env_size_x, CONFIG.env_size_y)
        num_actions = env.num_actions
        input_size = env.Lx
    else:  # vectorized
        env = gym.vector.AsyncVectorEnv(
            [
                lambda: GymSnakeEnv(size=(CONFIG.env_size_x, CONFIG.env_size_y))
                for _ in range(CONFIG.num_envs)
            ],
            context="fork",
        )
        num_actions = env.single_action_space.n  # type: ignore
        input_size = env.single_observation_space.spaces["agent"].high[0]  # type: ignore

    model = DQN(in_channels=1, num_actions=num_actions, input_size=input_size)
    model_target = DQN(in_channels=1, num_actions=num_actions, input_size=input_size)

    logging.debug(f"main, dqn_script: CONFIG.device: {CONFIG.device}")

    model = model.to(CONFIG.device)
    model_target = model_target.to(CONFIG.device)
    logging.debug(
        f"main, dqn_script: model.device: {model.device()}, model_target.device: {model_target.device()}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    loss_function = nn.HuberLoss()

    action_space = np.arange(num_actions)

    dqn_learning(env)
