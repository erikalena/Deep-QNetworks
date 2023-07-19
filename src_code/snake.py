import turtle
import time
import random
import sys
import pickle
from qlearning import *
from src_code.qnetworks import *
from src_code.cnn.agent import DQN
import torch



class game():

    def __init__(self, policy=None, model_path=None, nn_type=None, input_nodes=4, screen_width= 90, screen_height= 90, step= 10, delay= 0.5, border_size= -5):

        self.delay = delay
        self.score = 0
        self.high_score = 0
        self.step = step
        self.started = False
        self.reward_idx = 20 # initial random reward position

        self.border_size = border_size
        self.game_width = 200
        self.game_height = 200

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Set up the screen and game objects
        turtle.setworldcoordinates(0, self.game_height, self.game_width, 0)
        self.wn = turtle.Screen()
        self.snake = turtle.Turtle()
        
       
        self.body = []
        self.food = turtle.Turtle()
        self.header = turtle.Turtle()

        self.policy = policy
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.type = None
        self.input_nodes = input_nodes
        # if a model is specified, we use it to play the game
        if model_path != None and nn_type == 'cnn':
            self.Lx = self.Ly = int(self.game_width/20)
            model = DQN(in_channels =1, num_actions=4, input_size=self.Lx)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            model.to(self.device)
            self.model = model
            self.type = nn_type
        elif model_path != None:
            model = QLearnNN(self.input_nodes,4).to(self.device) #   QNetwork(4, 4, 42)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            self.model = model
        else:
            self.model = None


    def go_up(self):
        self.started = True
        self.snake.direction = "up"

    def go_down(self):
        self.tarted = True
        self.snake.direction = "down"

    def go_left(self):
        self.started = True
        self.snake.direction = "left"

    def go_right(self):
        self.tarted = True
        self.snake.direction = "right"

    def move_segments(self, prev_pos):
        # Move the end segments first in reverse order
        for index in range(len(self.body)-1, 0, -1):
            x = self.body[index-1].xcor() 
            y = self.body[index-1].ycor() 
    
            self.body[index].goto(x, y)

        # Move segment 0 to where the head is
        if len(self.body) > 0:
            x = self.snake.xcor()
            y = self.snake.ycor()
        
            self.body[0].goto(prev_pos[0], prev_pos[1])

    def move(self):
        wall = False # true if snake hits wall
        prev_pos = [self.snake.xcor(), self.snake.ycor()]

        # if conditions are not met, snake will not move
        if self.snake.direction == "down":
            if self.snake.ycor() >= self.screen_height: 
                self.snake.sety(0)
                wall = True
            else:
                y = self.snake.ycor()
                self.snake.sety(y + self.step)

        if self.snake.direction == "up" :
            if self.snake.ycor() <= 0:
                self.snake.sety(self.screen_height )
                wall = True
            else:
                y = self.snake.ycor()
                self.snake.sety(y - self.step)

        if self.snake.direction == "left":
            if self.snake.xcor() <= 0:
                self.snake.setx(self.screen_width )
                wall = True
            else:
                x = self.snake.xcor()
                self.snake.setx(x - self.step)

        if self.snake.direction == "right":
            if self.snake.xcor() >= self.screen_width:
                self.snake.setx(0)
                wall = True
            else:
                x = self.snake.xcor()
                self.snake.setx(x + self.step)
        
        # move the body segments
        self.move_segments(prev_pos)

        # check for collision with body
        self.check_body_collision()

        # check for collision with food
        self.check_food_collision()

        

        return wall

    def follow_policy(self):
        wall = False
        
        # x,y are discretized coordinates
        x = int(self.snake.xcor()/ self.step)
        y = int(self.snake.ycor()/ self.step)

        # find index of food
        self.reward_idx = int(self.food.ycor()/self.step)*self.step + int(self.food.xcor()/self.step)
        
        action = self.policy[y,x, int(self.reward_idx)]
        
        
        if action == 0:
            self.snake.direction = "down"
        elif action == 1:
            self.snake.direction = "up"
        elif action == 2:
            self.snake.direction = "right"
        elif action == 3:
            self.snake.direction = "left"
        
        wall = self.move()

        return wall

    def follow_model(self):
        wall = False
        
    
        if self.type == 'cnn':
            x = int(self.snake.xcor()*self.Lx/self.screen_width)
            y = int(self.snake.ycor()*self.Ly/self.screen_height)
            goal_x = int(self.food.xcor()*self.Lx/self.screen_width)
            goal_y = int(self.food.ycor()*self.Ly/self.screen_height)
            state = [y,x,goal_y, goal_x]
            input = self.get_image(state)
            input = torch.tensor(input).unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            # x,y are normalized coordinates
            x = (self.snake.xcor() + self.step)/ self.screen_width
            y = (self.snake.ycor() + self.step)/ self.screen_height
        
            goal_x = (self.food.xcor()+self.step)/ self.screen_width
            goal_y = (self.food.ycor()+self.step)/ self.screen_height

            state = [y,x,goal_y, goal_x]
            print('state', state)

            if self.input_nodes > len(state):
                #full_state = np.ones(self.input_nodes)*(-.01)
                full_state = np.zeros(self.input_nodes)
                full_state[:len(state)] = state 

                # if body is not empty
                if len(self.body) > 0:
                    for i in range(len(self.body)): # for each body segment
                        idx = len(state) + i*2
                        if idx < self.input_nodes:
                            full_state[idx] = (self.body[i].ycor() + self.step)/  self.screen_height
                            full_state[idx+1] = (self.body[i].xcor() + self.step)/  self.screen_width
                
                print(full_state)
            else:
                full_state = state
            input = torch.tensor(full_state).unsqueeze(0).to(self.device)

        action = self.model(input.float())
        action = torch.argmax(action).item()

        if action == 0:
            self.snake.direction = "down"
        elif action == 1:
            self.snake.direction = "up"
        elif action == 2:
            self.snake.direction = "right"
        elif action == 3:
            self.snake.direction = "left"
        
        wall = self.move()

        return wall

    def get_image(self, state):
        """
        Represent the game as an image, state input is a tuple of 4 elements
        (x,y,x_food,y_food)
        """
        image = np.zeros((self.Lx,self.Ly))
        head = state[:2]
        goal = state[2:]

        if (
            goal[0] >= 0 
            and goal[0] < self.Lx 
            and goal[1] >= 0 
            and goal[1] < self.Ly
        ):
            image[int(goal[0]), int(goal[1])] = .5

        if (
            head[0] >= 0 
            and head[0] < self.Lx 
            and head[1] >= 0 
            and head[1] < self.Ly
        ):
            image[int(head[0]), int(head[1])] = 1
        else:
            # if the agent is out of the world, it is dead and so we cancel the food and the body as well
            # this check is just for safety reasons, if we allow the snake to go through the walls
            # this should never happen
            image[int(goal[0]), int(goal[0])] = 0 
            
        for i in range(len(self.body)):
            if (self.body[i][0] >= 0 
                and self.body[i][0] < self.Lx 
                and self.body[i][1] >= 0 
                and self.body[i][1] < self.Ly
            ):
                image[int(self.body[i][0]), int(self.body[i][1])] = .1
            
        return image
    
    def check_food_collision(self):
        # Check for a collision with the food
        if self.snake.distance(self.food) <= self.step:

            # Move the food to a random spot
            x = random.randint(0, self.screen_width)
            y = random.randint(0, self.screen_height)

            # if the space is discretized and we are using a policy
            # we need to find the closest discretized coordinate
            if self.policy is not None:
                x = int(x/self.step)*self.step
                y = int(y/self.step)*self.step
            else:
                x = x - (x % self.step)
                y = y - (y % self.step)

            print('food', x,y)
            self.food.goto(y,x)

            # add a segment to the snake
            self.add_segment()

            self.score += 10

            if self.score > self.high_score:
                self.high_score = self.score
            
            self.update_header(self.score, self.high_score)

    def check_body_collision(self):
        # Check for head collision with the body segments
        for segment in self.body:
            #if segment.distance(self.snake) < self.step:
            if self.snake.xcor() == segment.xcor() and self.snake.ycor() == segment.ycor():
                self.end_game()
                print('died')

    def play(self):
        #self.snake.goto(40,60)
        # Main game loop
        while True:
            self.wn.update()

            if self.policy is not None:
                wall = self.follow_policy()  
            elif self.model is not None:
                wall = self.follow_model()
            else:
                wall = self.move()  

            # if we touched the wall, decide what to do
            if wall:
                pass

            time.sleep(self.delay)



    def create_environment(self):

        self.wn.title("Snake")
        self.wn.bgcolor("black")
        self.wn.setup(width=self.game_width, height=self.game_height)
        self.wn.tracer(0) # Turns off the screen updates

        # create a border aroung screen 
        border_pen = turtle.Turtle()
        border_pen.speed(0)
        border_pen.color("#0cab17")
        border_pen.penup()
        border_pen.setposition(self.border_size, self.border_size)
        border_pen.pendown()
        border_pen.pensize(3)
        for _ in range(4):
            border_pen.fd(self.screen_width +np.abs(self.border_size)*2)
            border_pen.rt(-90)


        border_pen.hideturtle()
        border_pen.penup()

        self.snake.speed(0)
        self.snake.shape("square")
        self.snake.color("#0cab17")
        self.snake.penup()
        self.snake.goto(0,0)
        self.snake.direction = "stop"

        self.food.speed(0)
        self.food.shape("circle")
        self.food.color("#a61654")
        self.food.penup()
        
        # find reward position
        y = 20
        x = 20
        self.reward_idx = int(y + x%self.step)
        self.food.goto(y,x)
        
        self.header.speed(0)
        self.header.shape("square")
        self.header.color("#0cab17")
        self.header.penup()
        self.header.hideturtle()
        self.header.goto(self.screen_width/2, -10)
        self.header.write("Score: 0 High Score: 0", align="center", font=("Calibri", 20, "normal"))

        return self.wn
    
    def update_header(self, score=0, high_score=0):
        self.header.clear()
        self.header.write("Score: {}  High Score: {}".format(score, high_score), align="center", font=("Calibri", 20, "normal"))

    def add_segment(self):
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("square")
        new_segment.color("#0cab17")
        new_segment.penup()
        self.body.append(new_segment)
        # set the position of the new segment
        # to the position of the last segment
        if len(self.body) == 1:
            new_segment.goto(self.snake.xcor(), self.snake.ycor())
        else:
            x = self.body[-2].xcor()
            y = self.body[-2].ycor()
            self.body[-1].goto(x, y)


    def end_game(self):
        time.sleep(.5)
        self.snake.goto(0,0)
        self.snake.direction = "stop"

        # hide the segments
        for segment in self.body:
            segment.goto(1000, 1000)

        self.body.clear()

        # Reset the score
        self.score = 0
        self.update_header(self.score, self.high_score)

        # reset the food
        self.food.goto(30,30)