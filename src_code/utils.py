

def add_parser_argument(parser, config):
    parser.add_argument(
        "--Lx", type=int, default=config.env_size_x, help="Size x of the environment"
    )
    parser.add_argument(
        "--Ly", type=int, default=config.env_size_y, help="Size y of the environment"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.max_num_episodes,
        help="Number of episodes",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.batch_size, help="Size of the batch"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=config.max_steps_per_episode,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--update_after",
        type=int,
        default=config.update_after_actions,
        help="Update after actions",
    )
    parser.add_argument(
        "--epsilon", type=float, default=config.epsilon, help="Epsilon value"
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=config.epsilon_min,
        help="Minimum epsilon value",
    )
    parser.add_argument(
        "--epsilon_max",
        type=float,
        default=config.epsilon_max,
        help="Maximum epsilon value",
    )
    parser.add_argument(
        "--update_target",
        type=int,
        default=config.update_target_network,
        help="Update target network",
    )
    parser.add_argument(
        "--epsilon_random_frames",
        type=int,
        default=config.epsilon_random_frames,
        help="Number of random frames",
    )
    parser.add_argument(
        "--epsilon_greedy_frames",
        type=float,
        default=config.epsilon_greedy_frames,
        help="Number of greedy frames",
    )
    parser.add_argument(
        "--output_log_dir",
        type=str,
        default=config.output_logdir,
        help="Output directory",
    )
    parser.add_argument(
        "--output_checkpoint_dir",
        type=str,
        default=config.output_checkpoint_dir,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--debug", type=bool, default=False, help="Debug mode (default: False)"
    )
    parser.add_argument(
        "--desc", type=str, default=config.description, help="Description of the run"
    )
    parser.add_argument(
        "--num_envs", type=int, default=config.num_envs, help="Number of environments"
    )
    return parser


class Statistics:
    def __init__(self):
        self.len = 0

    def shift(self, idx=1):
        pass

    def append(self):
        pass


class SeqStatistics(Statistics):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.points = []
        self.steps = []
        self.steps_per_point = None
        self.episode_rewards = []

    def shift(self, idx=1):
        self.losses = self.losses[idx:]
        self.points = self.points[idx:]
        self.steps = self.steps[idx:]
        self.episode_rewards = self.episode_rewards[idx:]

    def append(self, loss, points, steps, steps_per_point, episode_reward):
        self.losses.append(loss)
        self.points.append(points)
        self.steps.append(steps)
        self.episode_rewards.append(episode_reward)
        self.steps_per_point = steps_per_point
        self.len += 1


class VecStatistics(Statistics):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.episode_rewards = []

    def shift(self, idx=1):
        self.losses = self.losses[idx:]
        self.episode_rewards = self.episode_rewards[idx:]

    def append(self, loss, episode_reward):
        self.losses.append(loss)
        self.episode_rewards.append(episode_reward)
        self.len += 1