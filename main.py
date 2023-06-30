import turtle
import time
import random
import sys
import pickle
from qlearning import *
from snake2 import game

if __name__ == "__main__":
    

    mode = 0 # if no mode is specified, human play the game

    # read input arguments
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])

    if mode == 0:

        # initialize the game
        game = game()
        wn = game.create_environment()
        # Keyboard bindings
        wn.listen()

        wn.onkeypress(game.go_right, 'Right')
        wn.onkeypress(game.go_left, 'Left')
        wn.onkeypress(game.go_up, 'Up')
        wn.onkeypress(game.go_down, 'Down')

        print('here')
        # Main game loop
        game.play()

    elif mode == 1:
        # read policy from file and play the game using the policy
        with open('policy.pkl', 'rb') as f:
            agent = pickle.load(f)

            # if error, we exit
            if agent == None:
                print("Error: policy file not found")
                sys.exit()

        policy = agent.policy
        game = game(policy)
        wn = game.create_environment()
        game.play()