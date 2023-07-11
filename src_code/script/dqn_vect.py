import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
from src_code.qnetworks import ReplayBuffer
from src_code.deep_qnetworks_vectorized import DQN, SnakeEnv
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces


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
    num_envs: int = 3
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


##################################
# Utility functions
##################################


def get_image(state, body):
    """
    Represent the game as an image, state input is a tuple of 4 elements
    (x,y,x_food,y_food)
    """
    image = np.zeros((CONFIG.env_size_x, CONFIG.env_size_y))
    if state[2] >= 0 and state[2] < CONFIG.env_size_x and state[3] >= 0 and state[3] < CONFIG.env_size_y:
        image[int(state[2]), int(state[3])] = 1

    if state[0] >= 0 and state[0] < CONFIG.env_size_x and state[1] >= 0 and state[1] < CONFIG.env_size_y:
        image[int(state[0]), int(state[1])] = 1
    else:
        # if the agent is out of the world, it is dead and so we cancel the food as well
        # this check is just for safety reasons, if we allow the snake to go through the walls
        # this should never happen
        image[int(state[2]), int(state[3])] = 0

    for i in range(len(body)):
        if body[i][0] >= 0 and body[i][0] < CONFIG.env_size_x and body[i][1] >= 0 and body[i][1] < CONFIG.env_size_y:
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
            input = (
                torch.as_tensor(images, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(CONFIG.device)
            )

            qs = model(input).cpu().data.numpy()
            ret.append(np.argmax(qs))
    return ret


##################################
# Training
##################################
def train_step(
    states, actions, rewards, next_states, dones, bodies, new_bodies, discount
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

    # compute targets for Q-learning
    # the max Q-value of the next state is the target for the current state
    # the image to be fed to the network is a grey scale image of the world
    images = [
        get_image(next_state, new_body)
        for next_state, new_body in zip(next_states, new_bodies)
    ]
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
    images = [get_image(state, body) for state, body in zip(states, bodies)]
    input = (
        torch.as_tensor(np.array(images), dtype=torch.float32)
        .unsqueeze(1)
        .to(CONFIG.device)
    )
    qs = model(input)

    # for each state, we update ONLY the Q-value of the action that was taken

    # action_masks = F.one_hot(torch.as_tensor(np.array(actions)).long(), num_actions)
    action_masks = F.one_hot(actions.long(), num_actions)
    masked_qs = (action_masks * qs).sum(dim=-1)
    loss = loss_function(masked_qs, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def dqn_learning_vectorized(env, filename):
    # initialize the buffer, with a size of 100000, when it is full, it will remove the oldest element
    buffer = ReplayBuffer(size=100000, device=CONFIG.device)

    cur_frame = 0
    last_100_ep_rewards = []
    last_100_losses = []  #!! Added for tracking losses

    with open(filename, "w") as f:
        dict_json = {"configuration": CONFIG.__dict__}
        json.dump(dict_json, f, indent=4)

    epsilon = CONFIG.epsilon

    # env.start = np.array([0,0])
    pbar = tqdm(range(CONFIG.max_num_episodes))
    for episode in range(CONFIG.max_num_episodes):
        envs.reset()
        episode_reward = np.zeros(CONFIG.num_envs)

        timestep = 0
        accumulated_loss = 0
        while timestep < CONFIG.max_steps_per_episode:
            cur_frame += 1

            agent = np.stack(envs.get_attr("_agent_location"))
            target = np.stack(envs.get_attr("_target_location"))
            states = np.concatenate((agent, target), axis=1)
            bodies = list(envs.get_attr("body"))
            actions = select_epsilon_greedy_action(model, 1.0, states, bodies)

            next_states, rewards, dones, _, new_bodies = envs.step(actions)
            next_states = np.concatenate(
                (next_states["agent"], next_states["target"]), axis=1
            )
            episode_reward += rewards
            new_bodies = list(new_bodies["body"])

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
                loss = train_step(
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

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(episode_reward)

        if len(last_100_losses) == CONFIG.save_step:
            last_100_losses = last_100_losses[1:]
        last_100_losses.append(accumulated_loss / timestep)
        running_loss = np.mean(last_100_losses)

        running_reward = np.mean(last_100_ep_rewards)

        if episode + 1 % 100 == 0:
            """print(f'Episode {episode}/{max_num_episodes}. Epsilon: {epsilon:.3f}.'
            f' Reward in last 100 episodes: {running_reward:.2f}')"""
            logging.info(f"\n Saving results at episode {episode}")
            file = json.load(open(filename))
            file["episode_{}".format(episode)] = {
                "epsilon": epsilon,
                # "points": env.get_points(),
                # "steps" : timestep,
                # "steps_per_point": env.n_steps_per_point,
                "reward_mean": running_reward,
                "loss_mean": running_loss,
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
        if running_reward > 500:
            print("Solved at episode {}!".format(episode))
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
    parser.add_argument(
        "--num_envs", type=int, default=CONFIG.num_envs, help="Number of environments"
    )

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
        description="Vectorized DQN:" + args.desc,
    )

    logging.basicConfig(level=CONFIG.logging_level)
    logging.info(f"Start training with configuration: {CONFIG}")
    # make directory for checkpoints and results
    os.makedirs(CONFIG.output_logdir, exist_ok=True)
    os.makedirs(CONFIG.output_checkpoint_dir, exist_ok=True)

    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: SnakeEnv(size=(CONFIG.env_size_x, CONFIG.env_size_y))
            for _ in range(CONFIG.num_envs)
        ],
        context="forkserver",
    )

    # The first model makes the predictions for Q-values which are used to make a action.
    model = DQN(
        in_channels=1,
        num_actions=envs.single_action_space.n,
        input_size=envs.single_observation_space.spaces["agent"].high[0],
    )
    # The target model makes the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = DQN(
        in_channels=1,
        num_actions=envs.single_action_space.n,
        input_size=envs.single_observation_space.spaces["agent"].high[0],
    )

    model.to(CONFIG.device)
    model_target.to(CONFIG.device)

    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
    # huber loss
    loss_function = nn.HuberLoss()

    num_actions = envs.single_action_space.n
    action_space = np.arange(num_actions)
    envs.reset()
    filepath = f"{CONFIG.output_logdir}/{CONFIG.output_filename}"
    dqn_learning_vectorized(envs, filepath)
