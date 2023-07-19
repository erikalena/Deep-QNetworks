# Deep-QNetworks

![License MIT](https://img.shields.io/github/license/erikalena/Deep-QNetworks?style=for-the-badge) 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
## Purpose of the project

The aim of the project was to implement Deep QNetworks, following the description contained in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) and in particular to exploit it to make an agent learn how to play Snake game.

Instead of directly implementing DQN using CNNs, we progressively build up different models, starting from basic Qlearning algorithms and moving towards MLP, in order to deal with increasing complexity of the game. 
The code for all the trials we made is available, along with one simple version of Snake game, which can be played using all the models trained, from standard tabular methods to DQN.

## Content


## Requirements

The proejct was implemented using Python 3.9.5 and Pytorch 1.11.0.
The other libraries used are in the file requirements.txt.


## Usage

To play the game using models that implement the full version of snake, download the repository and move to *game* folder, then run the main script using the desired mode.
**Mode** can be: 
- human
- policy (simple Qlearning algorithm, just for simplest version of the game)
- mlp (a Qnetwork trained to play the full version of the game)
- cnn (full implementation of DQN to play snake)

```bash
cd game
python main.py [mode]
```
In results, already trained versions of different models are available, otherwise the user can train and load new ones.

### How to train MLP

Move in *mlp* directory and run the script, the results will be saved in the same directory, so that the training can be resumed after it terminates. 
In results file, the configuration of the network, along with all hyperparameters will be saved. Extra care is needed to properly tune all of these along with positive and negative rewards.

```bash
cd src_code/mlp
python train_mlp.py
```

### How to train CNN
From this folder run:
```bash
python -m src_code.script.dqn --output_log_dir results/cnn/<run-id>
                              --output_checkpoint_dir results/cnn/checkpoints/<run-id>
                              --load_checkpoint <path-checkpoint-to-load>
                              --num_episodes <number-of-episodes>
                              --max_steps_per_episode <max-steps-per-episode>
                              --save_step_per_episode <save-step-per-episode>
```
where:
- `<run-id>` is the name of the run, used to save the results and checkpoints
- `<path-checkpoint-to-load>` is the path to the checkpoint to load (if any). Specify it if you want to resume a previous training.
- `<number-of-episodes>` is the number of episodes to train the agent
- `<max-steps-per-episode>` is the maximum number of steps per episode
- `<save-step-per-episode>` is the number of episodes between each checkpoint save

In the results folder, a `log.json` file is created with all the configuration of the network and the hyperparameters used. It's also save statistics of the training each `<save-step-per-episode>` The checkpoints are saved in the `checkpoints` folder.
A folder `GIF` is also created, where the GIFs of the first and last 500 step of an episode are saved.

### GYM 


# TODO:
- [ ] ? Remove wall transparency
- [ ]  Add point/level system to vectorized version
- [ ]  check input cnn and epsilon decay
- [x] ? Vectorize the game (for speed)
- [x] Add DDQN 
- [ ] Add a `requirements.txt` file
