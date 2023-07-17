import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src_code.buffers import SeqReplayBuffer
from src_code.deep_qnetworks import DQN, SnakeEnv, SnakeAgent
from src_code.train import train_step

from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

import pickle
import logging
from dataclasses import dataclass
import argparse
import json
import os
import datetime


@dataclass
class Config:
    current_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    description: str = "Deep Q-Network Snake with gym and class agent, done after snake eats itself"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32  # Size of batch taken from replay buffer
    env_size_x: int = 20
    env_size_y: int = 20
    num_envs: int = 1
    max_steps_per_episode: int = 100000
    max_num_episodes: int = 10000
    deque_size: int = 100
    # epsilon
    epsilon_max: float = 1.0  # Maximum epsilon greedy parameter
    epsilon_min: float = 0.1  # Minimum epsilon greedy parameter
    eps_learning_rate: float = 0.01

    update_after_actions: int = 4  # Train the model after 4 actions
    update_target_network: int = 10000  # How often to update the target network
    epsilon_random_frames: int = 50000  # Number of frames for exploration
    epsilon_greedy_frames: float = 100000.0  # Number of frames for exploration
    buffer_size: int = 100000  # Size of the replay buffer

    output_filename: str = "log.json"
    output_logdir: str = "results/orfeo"
    output_checkpoint_dir: str = "checkpoint/orfeo"
    save_step: int = 50  # Save model every 100 episodes and log results
    logging_level: int = logging.DEBUG


CONFIG = Config()


def dqn_learning(CONFIG):
    # Save configuration
    filename: str = CONFIG.output_logdir + "/" + CONFIG.output_filename
    print(CONFIG)
    with open(filename, "w") as f:
        dict_json = {"configuration": CONFIG.__dict__}
        json.dump(dict_json, f, indent=4)

    env = SnakeEnv(size=(CONFIG.env_size_x, CONFIG.env_size_y))

    # model = DQN(
    #     in_channels=1, num_actions=env.action_space.n, input_size=CONFIG.env_size_x # type: ignore
    # )
    # model_target = DQN(
    #     in_channels=1, num_actions=env.action_space.n, input_size=CONFIG.env_size_x # type: ignore
    # )

    # model = model.to(CONFIG.device)
    # model_target = model_target.to(CONFIG.device)


    epsilon_decay = CONFIG.epsilon_max / (CONFIG.max_num_episodes * 0.5)

    snake_agent = SnakeAgent(
        learning_rate=CONFIG.eps_learning_rate,
        initial_epsilon=CONFIG.epsilon_max,
        final_epsilon=CONFIG.epsilon_min,
        epsilon_decay=epsilon_decay,
        num_actions=env.action_space.n, # type: ignore
        env=env,
        size=(CONFIG.env_size_x, CONFIG.env_size_y),
        device = CONFIG.device
    )

    buffer = SeqReplayBuffer(size=CONFIG.buffer_size, device=CONFIG.device)

    optimizer = torch.optim.Adam(snake_agent.model.parameters(), lr=0.00025)
    # huber loss
    loss_function = nn.HuberLoss()
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=CONFIG.deque_size)  # type: ignore
    
    cur_frame = 0
    fruits_eaten_per_episode = []
    for episode in tqdm(range(CONFIG.max_num_episodes)):
        obs, info = env.reset()
        terminated = False
        timestep = 0
        while not terminated:
            cur_frame += 1
            action = snake_agent.get_action(obs, info)
            new_obs, reward, done, _, new_info = env.step(action)
            terminated = done or (timestep > CONFIG.max_steps_per_episode)

            # Save actions and states in replay buffer
            buffer.add(obs, action, reward, new_obs, done, info, new_info)

            # Update obs and info
            obs = new_obs
            info = new_info

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
                loss = train_step(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    bodies,
                    new_bodies,
                    snake_agent,
                    loss_function,
                    optimizer,
                    CONFIG.device
                )

            # Update target network every update_target_network steps.
            if cur_frame % CONFIG.update_target_network == 0:
                snake_agent.model_target.load_state_dict(snake_agent.model.state_dict())

            timestep += 1
        if episode>CONFIG.epsilon_random_frames:
            snake_agent.decay_epsilon()

        fruits_eaten_per_episode.append(env.eaten_fruits)
        
        if ((episode + 1) % CONFIG.save_step) == 0:
            # write on file current average reward
            metrics = {
                "return_queue": env.return_queue,
                "length_queue": env.length_queue,
                "training_error": snake_agent.training_error,
                "epsilon": snake_agent.epsilon,
            }
            file = json.load(open(filename))
            file["episode_{}".format(episode)] = {
                "training_error": str(np.mean(snake_agent.training_error)),
                "mean_eaten": np.mean(fruits_eaten_per_episode),
                "eatens": env.eaten_fruits,
                "epsilon": str(snake_agent.epsilon),
                }
            json.dump(file, open(filename, "w"), indent=4)
            fruits_eaten_per_episode = []
            with open(
                CONFIG.output_logdir + "/metrics_{}".format(episode), "wb"
            ) as handle:
                pickle.dump(metrics, handle)
            # do we want to save it every 100 episodes? dunno it's up to you
            torch.save(
                snake_agent.model.state_dict(),
                CONFIG.output_checkpoint_dir + "/model_{}".format(episode),
            )
            # save_fig(env, snake_agent, episode)

        # Condition to consider the task solved
        if np.mean(env.return_queue) > 500:  # type: ignore
            print("Solved at episode {}!".format(episode))
            break


def save_fig(env, snake_agent, episode):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(snake_agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.savefig(CONFIG.output_logdir + "/training_metrics_{}.png".format(episode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Q-Network Snake")
    parser.add_argument(
        "--output_log_dir",
        type=str,
        default=CONFIG.output_logdir,
        help="Output directory",
    )
    parser.add_argument(
        "--output_checkpoint_dir",
        type=str,
        default=CONFIG.output_checkpoint_dir,
        help="Output directory for checkpoints",
    )
    args = parser.parse_args()

    CONFIG = Config(
        output_logdir=args.output_log_dir,
        output_checkpoint_dir=args.output_checkpoint_dir,
    )

    os.makedirs(CONFIG.output_logdir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)
    dqn_learning(CONFIG)
