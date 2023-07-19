import turtle
import time
import random
import sys
import pickle
from qlearning import *
from snake import game


def print_usage():
    print("Usage: python main.py [mode]")
    print("""
          mode: 
          human : human play the game 
          policy: read tabular policy 
          mlp: read MLP policy 
          cnn: read CNN policy
          """
        )

if __name__ == "__main__":
    
    mode = None
    # read input arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['human', 'policy', 'mlp', 'cnn']:
        mode = sys.argv[1]
    else:
        print_usage()

    if mode == 'human':

        # initialize the game
        game = game(policy=None, screen_width=90, screen_height=90, step=2.5, delay=0.2)
        wn = game.create_environment()
        # Keyboard bindings
        wn.listen()

        wn.onkeypress(game.go_right, 'Right')
        wn.onkeypress(game.go_left, 'Left')
        wn.onkeypress(game.go_up, 'Up')
        wn.onkeypress(game.go_down, 'Down')

        # Main game loop
        game.play()

    elif mode == 'policy':
        # read policy from file and play the game using the policy
        with open('../checkpoint/policy.pkl', 'rb') as f:
            agent = pickle.load(f)

            # if error, we exit
            if agent == None:
                print("Error: policy file not found")
                sys.exit()

        policy = agent.policy
        game = game(policy, step=10, delay=0.5)
        wn = game.create_environment()
        game.play()

    elif mode == 'mlp':
        game = game(step=5, delay=0.1, model_path='model_14600.pth', input_nodes=24)#'../checkpoint/qlearnNN.pt') 
        wn = game.create_environment()
        game.play()

    elif mode == 'cnn':
        game = game(step=3, delay=0.2, model_path='../checkpoint/17248/model_2000.pth', nn_type='cnn')
        wn = game.create_environment()
        game.play()