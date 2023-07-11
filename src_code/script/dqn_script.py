from src_code.qnetworks import ReplayBuffer
from src_code.deep_qnetworks import DQN, SnakeEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import argparse
import logging
from tqdm import tqdm
import json
import os
import datetime


@dataclass
class Config:
    current_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32  # Size of batch taken from replay buffer
    env_size_x: int = 20
    env_size_y: int = 20
    max_steps_per_episode: int = 10000
    max_num_episodes: int = 5000
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
    output_logdir: str = "results"
    output_checkpoint_dir: str = "checkpoints"
    save_step: int = 50  # Save model every 100 episodes and log results
    logging_level: int = logging.INFO
    description: str = "Deep Q-Network Snake, done after snake eats itself"


CONFIG = Config()
# logging


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
    input = (
        torch.as_tensor(np.array(images), dtype=torch.float32)
        .unsqueeze(1)
        .to(CONFIG.device)
    )
    max_next_qs = model_target(input).max(-1).values

    # if the next state is terminal, then the Q-value is just the reward
    # otherwise, we add the discounted max Q-value of the next state
    target = rewards + (1.0 - dones) * discount * max_next_qs

    # then to compute the loss, we also need the Q-value of the current state
    images = [env.get_image(state) for state in states]
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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def dqn_learning(env, filename):
    # initialize the buffer, with a size of 100000, when it is full, it will remove the oldest element
    buffer = ReplayBuffer(size=100000, device=CONFIG.device)

    cur_frame = 0
    last_100_ep_rewards = []  #!! Why do not reset to each episode?
    last_100_losses = []  #!! Added for tracking losses
    last_100_points = []  #!! Added for tracking points
    last_100_steps = []  #!! Added for tracking steps
    last_100_steps_per_point = []  #!! Added for tracking steps per point

    # print all configuration of file
    # open file
    # write all configuration in the json file
    with open(filename, "w") as f:
        dict_json = {"configuration": CONFIG.__dict__}
        json.dump(dict_json, f, indent=4)

    epsilon = CONFIG.epsilon

    env.start = np.array([0, 0])
    pbar = tqdm(range(CONFIG.max_num_episodes), desc="Run episodes")
    for episode in range(CONFIG.max_num_episodes):
        env.reset()
        episode_reward = 0

        # state is a tuple of 4 values made of starting position and goal position
        # start of an episode is always [0,0] for snake and a random position for goal
        start_x = env.start[0]
        start_y = env.start[1]
        goal_x = np.random.randint(0, env.Lx)
        goal_y = np.random.randint(0, env.Ly)

        state = [start_x, start_y, goal_x, goal_y]

        # done = False
        timestep = 0
        accumulated_loss = 0
        while timestep < CONFIG.max_steps_per_episode:
            cur_frame += 1

            state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(CONFIG.device)
            action = env.select_epsilon_greedy_action(model, state_in, epsilon)

            next_state, reward, done = env.single_step(state, action)
            episode_reward += reward

            # Save actions and states in replay buffer
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            cur_frame += 1

            # Train neural network.
            if (
                len(buffer) > CONFIG.batch_size
                and cur_frame % CONFIG.update_after_actions == 0
            ):
                states, actions, rewards, next_states, dones = buffer.sample(
                    CONFIG.batch_size
                )
                loss = train_step(
                    states, actions, rewards, next_states, dones, discount=0.99
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

        if len(last_100_ep_rewards) == CONFIG.save_step:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(episode_reward)
        running_reward = np.mean(last_100_ep_rewards)

        if len(last_100_losses) == CONFIG.save_step:
            last_100_losses = last_100_losses[1:]
        last_100_losses.append(accumulated_loss / timestep)
        running_loss = np.mean(last_100_losses)

        if len(last_100_points) == CONFIG.save_step:
            last_100_points = last_100_points[1:]
        last_100_points.append(env.get_points())
        running_points = np.mean(last_100_points)

        if len(last_100_steps) == CONFIG.save_step:
            last_100_steps = last_100_steps[1:]
        last_100_steps.append(timestep)
        running_steps = np.mean(last_100_steps)

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
                "reward_mean": running_reward,
                "loss_mean": running_loss,
                "points_mean": running_points,
                "steps_mean": running_steps,
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
        if running_reward > 500:
            logging.info("Solved at episode {}!".format(episode))
            break
        pbar.update(1)


if __name__ == "__main__":
    # read input arguments
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Deep Q-Network Snake")
    parser.add_argument(
        "--Lx", type=int, default=CONFIG.env_size_x, help="Size x of the environment"
    )
    parser.add_argument(
        "--Ly", type=int, default=CONFIG.env_size_y, help="Size y of the environment"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=CONFIG.max_num_episodes,
        help="Number of episodes",
    )
    parser.add_argument(
        "--batch_size", type=int, default=CONFIG.batch_size, help="Size of the batch"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=CONFIG.max_steps_per_episode,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--update_after",
        type=int,
        default=CONFIG.update_after_actions,
        help="Update after actions",
    )
    parser.add_argument(
        "--epsilon", type=float, default=CONFIG.epsilon, help="Epsilon value"
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=CONFIG.epsilon_min,
        help="Minimum epsilon value",
    )
    parser.add_argument(
        "--epsilon_max",
        type=float,
        default=CONFIG.epsilon_max,
        help="Maximum epsilon value",
    )
    parser.add_argument(
        "--update_target",
        type=int,
        default=CONFIG.update_target_network,
        help="Update target network",
    )
    parser.add_argument(
        "--epsilon_random_frames",
        type=int,
        default=CONFIG.epsilon_random_frames,
        help="Number of random frames",
    )
    parser.add_argument(
        "--epsilon_greedy_frames",
        type=float,
        default=CONFIG.epsilon_greedy_frames,
        help="Number of greedy frames",
    )
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
        "--debug", type=bool, default=False, help="Debug mode (default: False)"
    )
    parser.add_argument(
        "--desc", type=str, default=CONFIG.description, help="Description of the run"
    )

    args = parser.parse_args()

    CONFIG = Config(
        current_time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        logging_level=logging.DEBUG if args.debug else CONFIG.logging_level,
        env_size_x=args.Lx,
        env_size_y=args.Ly,
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
        description="Serial DQN:" + args.desc,
    )

    logging.basicConfig(level=CONFIG.logging_level)

    logging.info(f"Start training with configuration: {CONFIG}")
    # make directory for checkpoints and results
    os.makedirs(CONFIG.output_logdir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)

    env = SnakeEnv(CONFIG.env_size_x, CONFIG.env_size_y)

    model = DQN(in_channels=1, num_actions=env.num_actions, input_size=env.Lx)
    model_target = DQN(in_channels=1, num_actions=env.num_actions, input_size=env.Lx)

    logging.debug(f"main, dqn_script: CONFIG.device: {CONFIG.device}")

    model = model.to(CONFIG.device)
    model_target = model_target.to(CONFIG.device)
    logging.debug(
        f"main, dqn_script: model.device: {model.device()}, model_target.device: {model_target.device()}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    loss_function = nn.HuberLoss()

    num_actions = env.num_actions
    action_space = np.arange(num_actions)
    filepath = f"{CONFIG.output_logdir}/{CONFIG.output_filename}"
    # filename = "dqn_results.txt"
    dqn_learning(env, filepath)
