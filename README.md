# Deep-QNetworks

## Purpose of the project

The aim of the project was to implement Deep QNetworks, following the description contained in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) and in particular to exploit it to make an agent learn how to play Snake game.

Instead of directly implementing DQN using CNNs, we progressively build up different models, starting from basic Qlearning algorithms and moving towards MLP, in order to deal with increasing complexity of the game. 
The code for all the trials we made is available, along with one simple version of Snake game, which can be played using all the models trained, from standard tabular methods to DQN.


## Requirements

The proejct was implemented using Python 3.9.5 and Pytorch 1.11.0.
The other libraries used are in the file requirements.txt.


## Usage

To play the game using models that implement the full version of snake, download th repository and move to *src_code* folder, then run the main script using MLP or CNNs as you prefer.

```bash
cd src_code
python main.py [mlp | cnn]
```

An already trained version is available for both the models, but new versions can be trained as well.

### How to train MLP



### How to train CNN


### GYM 


# TODO:
- [ ] ? Remove wall transparency
- [ ]  Add point/level system to vectorized version
- [ ]  check input cnn and epsilon decay
- [x] ? Vectorize the game (for speed)
- [x] Add DDQN 
- [ ] Add a `requirements.txt` file
