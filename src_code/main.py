import turtle
import time
import random
import sys
import pickle
from qlearning import *
from snake import game




if __name__ == "__main__":
    

    mode = 0 # if no mode is specified, human play the game

    # read input arguments
    if len(sys.argv) > 1:
        mode = int(sys.argv[1])

    if mode == 0:

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

    elif mode == 1:
        # read policy from file and play the game using the policy
        with open('policy.pkl', 'rb') as f:
            agent = pickle.load(f)

            # if error, we exit
            if agent == None:
                print("Error: policy file not found")
                sys.exit()

        policy = agent.policy
        game = game(policy, step=10, delay=0.5)
        wn = game.create_environment()
        game.play()

    elif mode == 2:
        game = game(step=3, delay=0.2, model_path='qlearnNN.pt') #model_path='qlearn_snake_model.pt')
        wn = game.create_environment()
        game.play()

    elif mode == 3:
        game = game(step=3, delay=0.2, model_path='../checkpoint/17248/model_2000.pth', type='cnn')
        wn = game.create_environment()
        game.play()