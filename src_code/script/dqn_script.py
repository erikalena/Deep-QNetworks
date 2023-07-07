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


@dataclass
class Config:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128  # Size of batch taken from replay buffer
    env_size_x: int = 20
    env_size_y: int = 20
    max_steps_per_episode: int = 5000
    max_num_episodes: int = 5000
    update_after_actions: int = 4
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


CONFIG = Config()
# logging
logging.basicConfig(level=logging.INFO)


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

            # Update target network every update_target_network steps.
            if cur_frame % CONFIG.update_target_network == 0:
                model_target.load_state_dict(model.state_dict())

            timestep += 1

            if timestep > CONFIG.epsilon_random_frames:
                epsilon -= (
                    CONFIG.epsilon_max - CONFIG.epsilon_min
                ) / CONFIG.epsilon_greedy_frames
                epsilon = max(CONFIG.epsilon, CONFIG.epsilon_min)

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(episode_reward)
        running_reward = np.mean(last_100_ep_rewards)

        if len(last_100_losses) == 100:
            last_100_losses = last_100_losses[1:]
        last_100_losses.append(loss.item())
        running_loss = np.mean(last_100_losses)

        if episode % 100 == 0:
            """print(f'Episode {episode}/{max_num_episodes}. Epsilon: {epsilon:.3f}.'
            f' Reward in last 100 episodes: {running_reward:.2f}')"""
            file = json.load(open(filename))
            file["episode_{}".format(episode)] = {
                "epsilon": epsilon,
                "reward": running_reward,
                "loss": running_loss,
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
    parser.add_argument("--size", type=int, default=20, help="Size of the environment")
    parser.add_argument(
        "--episodes", type=int, default=10000, help="Number of episodes"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Size of the batch")
    parser.add_argument(
        "--max_steps", type=int, default=5000, help="Max steps per episode"
    )
    parser.add_argument(
        "--update_after", type=int, default=4, help="Update after actions"
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon value")
    parser.add_argument(
        "--epsilon_min", type=float, default=0.1, help="Minimum epsilon value"
    )
    parser.add_argument(
        "--epsilon_max", type=float, default=1.0, help="Maximum epsilon value"
    )
    parser.add_argument(
        "--update_target", type=int, default=10000, help="Update target network"
    )
    parser.add_argument(
        "--epsilon_random_frames",
        type=int,
        default=50000,
        help="Number of random frames",
    )
    parser.add_argument(
        "--epsilon_greedy_frames",
        type=float,
        default=100000.0,
        help="Number of greedy frames",
    )
    parser.add_argument(
        "--output_log_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--output_checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Update the configuration values
    CONFIG.env_size_x = args.size
    CONFIG.env_size_y = args.size
    CONFIG.max_num_episodes = args.episodes
    CONFIG.batch_size = args.batch_size
    CONFIG.max_steps_per_episode = args.max_steps
    CONFIG.update_after_actions = args.update_after
    CONFIG.epsilon = args.epsilon
    CONFIG.epsilon_min = args.epsilon_min
    CONFIG.epsilon_max = args.epsilon_max
    CONFIG.update_target_network = args.update_target
    CONFIG.epsilon_random_frames = args.epsilon_random_frames
    CONFIG.epsilon_greedy_frames = args.epsilon_greedy_frames
    CONFIG.output_logdir = args.output_log_dir
    CONFIG.output_checkpoint_dir = args.output_checkpoint_dir

    logging.info(f"Start training with configuration: {CONFIG}")
    # make directory for checkpoints and results
    os.makedirs(CONFIG.output_logdir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)

    env = SnakeEnv(CONFIG.env_size_x, CONFIG.env_size_y)

    model = DQN(in_channels=1, num_actions=env.num_actions, input_size=env.Lx)
    model_target = DQN(in_channels=1, num_actions=env.num_actions, input_size=env.Lx)

    model.to(CONFIG.device)
    model_target.to(CONFIG.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    loss_function = nn.HuberLoss()

    num_actions = env.num_actions
    action_space = np.arange(num_actions)
    filepath = f"{CONFIG.output_logdir}/{CONFIG.output_filename}"
    # filename = "dqn_results.txt"
    dqn_learning(env, filepath)
