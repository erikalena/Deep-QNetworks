import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy

from src_code.cnn.buffers import SeqReplayBuffer
from src_code.cnn.agent import DQN, SnakeAgent
from src_code.cnn.environment import SnakeEnv
from src_code.cnn.train import d_train_step

from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

import pickle
import logging
from dataclasses import dataclass, field
import argparse
import json
import os
import io
import datetime
import logging

from PIL import Image
from typing import Dict, List, Tuple, Union


@dataclass
class Config:
    current_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    description: str = (
        "Deep Q-Network Snake with gym and class agent, done after snake eats itself"
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32  # Size of batch taken from replay buffer
    env_size_x: int = 10
    env_size_y: int = 10
    num_envs: int = 1
    max_steps_per_episode: int = 30000
    max_num_episodes: int = 20000
    deque_size: int = 100

    # epsilon
    epsilon_max: float = 1.0  # Maximum epsilon greedy parameter
    epsilon_min: float = 0.2  # Minimum epsilon greedy parameter

    no_back = False
    done_on_collision: bool = False
    update_after_actions: int = 4  # Train the model after 4 actions
    update_target_network: int = 10000  # How often to update the target network
    epsilon_random_frames: int = 100000  # Number of frames for exploration
    eps_decay_frames: int = 10000
    buffer_size: int = 100000  # Size of the replay buffer
    eps_decay_rate: float = 0.001
    reward: Dict[str, int] = field(
        default_factory=lambda: {"eat": 40, "dead": -1, "step": 0}
    )

    output_filename: str = "log.json"
    output_logdir: str = "results/orfeo"
    output_checkpoint_dir: str = "checkpoint/orfeo"
    save_step: int = 100  # Save model every 100 episodes and log results
    logging_level: int = logging.DEBUG
    load_checkpoint: str =  "checkpoint/18839/model_8299"


CONFIG = Config()


def create_gif_from_plt_images(image_list, output_path, duration=200):
    temp_image_folder = "temp_images"
    os.makedirs(temp_image_folder, exist_ok=True)

    images = []
    for i, img_data in enumerate(image_list):
        img_path = os.path.join(temp_image_folder, f"temp_{i}.png")
        plt.imshow(img_data)
        plt.axis("off")
        with open(img_path, "wb") as file:
            plt.savefig(file, bbox_inches="tight", pad_inches=0)
        plt.clf()
        images.append(img_path)

    gif_images = [Image.open(img) for img in images]

    gif_images[0].save(
        output_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=duration,
        loop=0,
    )

    # Clean up temporary images
    for img_path in images:
        os.remove(img_path)


def ddqn_learning(CONFIG):
    # Save configuration
    filename: str = CONFIG.output_logdir + "/" + CONFIG.output_filename
    print(CONFIG)
    with open(filename, "w") as f:
        dict_json = {"configuration": CONFIG.__dict__}
        json.dump(dict_json, f, indent=4)

    env = SnakeEnv(size=(CONFIG.env_size_x, CONFIG.env_size_y), config=CONFIG)

    epsilon_decay = (CONFIG.epsilon_max / (CONFIG.max_num_episodes * 0.5)) * 100

    snake_agent = SnakeAgent(
        initial_epsilon=CONFIG.epsilon_max,
        final_epsilon=CONFIG.epsilon_min,
        epsilon_decay=CONFIG.eps_decay_rate,
        num_actions=env.action_space.n,  # type: ignore
        env=env,
        size=(CONFIG.env_size_x, CONFIG.env_size_y),
        device=CONFIG.device,
    )

    buffer = SeqReplayBuffer(size=CONFIG.buffer_size, device=CONFIG.device)

    optimizer = torch.optim.Adam(snake_agent.model.parameters(), lr=0.00025)
    if CONFIG.load_checkpoint is not None:
        optimizer = snake_agent.load_model(CONFIG.load_checkpoint, optimizer=optimizer)
    # huber loss
    loss_function = nn.HuberLoss()
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=CONFIG.deque_size)  # type: ignore

    cur_frame = 0
    fruits_eaten_per_episode = []
    pbar = tqdm(total=CONFIG.max_num_episodes)
    tot_step = 0
    for episode in range(CONFIG.max_num_episodes):
        obs, info = copy.deepcopy(env.reset())
        terminated = False
        timestep = 0
        frames = []
        rewards_per_episode = []
        n_step_per_episode = []
        while not terminated:
            cur_frame += 1
            tot_step += 1
            action = snake_agent.get_action(obs, info)
            new_obs, reward, done, selected_action, new_info = env.step(action)
            terminated = done or (timestep > CONFIG.max_steps_per_episode)

            #
            rewards_per_episode.append(reward)
            tmp_state = np.concatenate((obs["agent"], obs["target"]))
            tmp_body = info["body"]

            if timestep < 1002 or timestep > CONFIG.max_steps_per_episode - 502:
                frame = snake_agent.get_image(tmp_state, tmp_body)
                frames.append(frame)
            # Save actions and states in replay buffer
            buffer.add(obs, selected_action, reward, new_obs, done, info, new_info)

            # Update obs and info
            obs = copy.deepcopy(new_obs)
            info = copy.deepcopy(new_info)

            cur_frame += 1

            if tot_step > CONFIG.epsilon_random_frames :
                if tot_step % CONFIG.eps_decay_frames == 0:
                    snake_agent.decay_epsilon()

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
                loss = d_train_step(
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
                    CONFIG.device,
                )

            # Update target network every update_target_network steps.
            if cur_frame % CONFIG.update_target_network == 0:
                snake_agent.model_target.load_state_dict(snake_agent.model.state_dict())

            timestep += 1
            # if env.eaten_fruits == 10:
            n_step_per_episode.append(timestep)
            #     break
        # if episode>CONFIG.epsilon_random_frames:
        # snake_agent.decay_epsilon()

        fruits_eaten_per_episode.append(env.eaten_fruits)

        if ((episode + 1) % CONFIG.save_step) == 0:
            # Save episode in a gif
            os.makedirs(CONFIG.output_logdir + "/GIF", exist_ok=True)
            try:
                if timestep > 1000:
                    output_gif = CONFIG.output_logdir + "/GIF/game_ep_{}_1.gif".format(
                        episode
                    )
                    create_gif_from_plt_images(frames[0:500], output_gif, duration=200)
                    output_gif = CONFIG.output_logdir + "/GIF/game_ep_{}_2.gif".format(
                        episode
                    )
                    create_gif_from_plt_images(frames[-500:], output_gif, duration=200)
                else:
                    output_gif = CONFIG.output_logdir + "/GIF/game_{}.gif".format(
                        episode
                    )
                    create_gif_from_plt_images(frames, output_gif, duration=200)
            except:
                pass

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
                "mean_step": np.mean(n_step_per_episode),
                "mean_eaten": np.mean(fruits_eaten_per_episode),
                "mean_reward": np.mean(rewards_per_episode),
                "eatens": env.eaten_fruits,
                "epsilon": str(snake_agent.epsilon),
            }
            json.dump(file, open(filename, "w"), indent=4)
            fruits_eaten_per_episode = []
            os.makedirs(CONFIG.output_logdir + "/metrics", exist_ok=True)
            with open(
                CONFIG.output_logdir + "/metrics/metrics_{}".format(episode), "wb"
            ) as handle:
                pickle.dump(metrics, handle)
            # do we want to save it every 100 episodes? dunno it's up to you
            snake_agent.save_model(
                CONFIG.output_checkpoint_dir + "/model_{}".format(episode),
                optimizer=optimizer,
            )
            # save_fig(env, snake_agent, episode)

        # Condition to consider the task solved
        if np.mean(env.return_queue) > 500:  # type: ignore
            print("Solved at episode {}!".format(episode))
            break
        pbar.update(1)
        pbar.set_description("Epsilon: {}".format(snake_agent.epsilon))


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
        np.convolve(
            np.array(snake_agent.training_error), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )
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
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=CONFIG.load_checkpoint,
        help="Load checkpoint",
    )
    args = parser.parse_args()

    CONFIG = Config(
        output_logdir=args.output_log_dir,
        output_checkpoint_dir=args.output_checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
    )
    logging.basicConfig(level=CONFIG.logging_level)
    os.makedirs(CONFIG.output_logdir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)
    ddqn_learning(CONFIG)
