import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
import pickle
import copy

plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams['figure.dpi'] = 100 
plt.rcParams['font.size'] = 6



class GridWorldEnv():
    def __init__(self, Lx, Ly, start = None, end = None):
        
        # World shape
        self.Ly, self.Lx = Lx, Ly
 
        # policy is a matrix of size Ly x Lx x S 
        # where S is the number of states
        self.policy = np.zeros((self.Ly, self.Lx, self.Ly*self.Lx ))
        self.values = np.zeros((self.Ly, self.Lx, self.Ly*self.Lx ))

        # start and end positions
        self.start = [0,0] if start is None else start
        self.end = None if end is None else end

        # generate random positions for the reward
        self.final_pos = [np.random.randint(1, self.Ly), np.random.randint(1, self.Lx) ] if end is None else end
        
        # generate world instance
        reward = 10
        self.World = self.new_instance(Lx, Ly, self.final_pos, reward)
        
        # Keeps track of current state
        self.current_state = self.start
        
        # Keeps track of terminal state
        self.done = False

        self.gamma = 1.0
        self.lr = 0.15
        self.lr_0 = copy.deepcopy(self.lr)

        self.epsilon = 0.6
        self.epsilon_0 = copy.deepcopy(self.epsilon)

        #Actions = [Down,   Up,  Right,Left] 
        self.actions = np.array([[1,0],[-1,0],[0,1],[0,-1]])

        self.n_episodes = 1000
    
    def new_instance(self, Lx, Ly, goal, rewards):
        World = np.zeros((Ly,Lx))
        World[goal[0], goal[1]] = rewards
        return World

        
    def reset(self):
        """
        Resets the GridWorld to the starting position.
        """
        # Reset the environment to initial state
        self.current_state = self.start
        self.done = False
        
    def step(self, A, state = None):
        """
        Evolves the environment given action A and current state.
        """
        # Check if action A is in proper set
        assert A in np.array([[1,0],[-1,0],[0,1],[0,-1]]) 
        S = self.current_state if state is None else state
        S_new = S + A

        
        # Always a penalty for moving -> want to find the shortest path!
        reward = -1

        # If I go out of the world, we enter from the other side
        if (S_new[0] == self.Ly):
            S_new[0] = 0
        elif (S_new[0] == -1):
            S_new[0] = self.Ly - 1
        elif (S_new[1] == self.Lx):
            S_new[1] = 0
        elif (S_new[1] == -1):
            S_new[1] = self.Lx - 1
        
        elif np.all(S_new == self.end):
            self.done = True         
        
        # Save in memory new position
        self.current_state = S_new
            
        return S_new, reward, self.done

    def learn_policy(self, save=True):
        """
        Learn final policy running QLearning algorithm for
        a suitable number of episodes
        """
        
        # for each possible position of the reward
        for i in range(self.Ly):
            for j in range(self.Lx):

                # set the reward position
                self.end = np.array([i,j])

                # choose as starting position one which is far from the reward
                self.start = self.end + np.array([self.Ly//2, self.Lx//2])
                self.start[0] = self.start[0] % self.Ly
                self.start[1] = self.start[1] % self.Lx
                
                # Initialize 
                QLearning = Qlearning_TDControl(space_size=self.World.shape, action_size=4, gamma=self.gamma, lr_v=self.lr)

                # save number of steps necessary to complete an episode
                n_steps = np.zeros(self.n_episodes)
                greedy_steps = 200

                for e in range(self.n_episodes):
                    done = False

                    # reset the environment
                    self.reset()

                    s = self.current_state
                    a = QLearning.get_action_epsilon_greedy(s, self.epsilon)
                    act = self.actions[a]
                    

                    while not done:
                        # Evolve one step
                        new_s, r, done = self.step(act)
                                                    
                        # Choose new action index
                        new_a = QLearning.get_action_epsilon_greedy(new_s, self.epsilon)

                        # (Corresponding action to index)
                        act = self.actions[new_a]
                        # Single update with (S, A, R', S')
                        QLearning.single_step_update(s, a, r, new_s, done)
                        
                        a = new_a
                        s = new_s
                        n_steps[e] += 1

                        if n_steps[e] > greedy_steps:
                            # update learning rate
                            self.lr_v = self.lr_0/(1 + 0.003*(n_steps[e] - greedy_steps)**0.75)
                            # update epsilon
                            self.epsilon = self.epsilon_0/(1. + 0.005*(n_steps[e] - greedy_steps)**1.05)

                self.policy[:,:,i*self.Ly+j] = QLearning.greedy_policy().astype(int)    

                for m in range(self.Ly):
                    for l in range(self.Lx):
                        self.values[m,l,i*self.Ly+j] = QLearning.Qvalues[m,l,self.policy[m,l,i*self.Ly+j].astype(int)]        
            
            
            print("Policy for goal in position ({},*) learned".format(i))

        # save the policy and values in pickle file
        if save:
            with open('policy.pkl', 'wb') as f:
                pickle.dump(self, f)

        return n_steps

        
    def get_qvalues(self, qlearn, n_episodes, end, start=None):
        """
        Get the Qvalues resulting from the execution of Qlearning algorithm for a 
        given starting point and an end point, run for a small number of episodes.
        
        """

        # set the reward position
        self.end = np.array([end[0], end[1]])

        if start is None:
            # choose as starting position one which is far from the reward
            self.start = self.end + np.array([self.Ly//2, self.Lx//2])
        else:
            self.start = start
        self.start[0] = self.start[0] % self.Ly
        self.start[1] = self.start[1] % self.Lx
        
        

        for _ in range(n_episodes):
            done = False

            # reset the environment
            self.reset()

            s = self.current_state
            a = qlearn.get_action_epsilon_greedy(s, self.epsilon)
            act = self.actions[a]
            
            while not done:
                # Evolve one step
                new_s, r, done = self.step(act)
                                            
                # Choose new action index
                new_a = qlearn.get_action_epsilon_greedy(new_s, self.epsilon)
                act = self.actions[new_a]

                # Single update with (S, A, R', S')
                qlearn.single_step_update(s, a, r, new_s, done)
                
                a = new_a
                s = new_s

        return qlearn
        

    def render(self):
        Ly, Lx = self.World.shape

        fig, ax = plt.subplots()
        im = ax.imshow(self.World, cmap='Set3')
        
        goal = np.where(self.World > 0.0)
        
        # Loop over data dimensions and create text annotations.
        for i in range(Lx):
            for j in range(Ly):
                if np.logical_and(goal[0]==j,goal[1]==i).any():
                    text = ax.text(i,j, 'G{}'.format(self.World[j,i]), ha="center", va="center", color="black")
                elif self.start[1]!=j or self.start[0]!=i:
                    text = ax.text(i,j, '{}'.format(self.World[j,i]), ha="center", va="center", color="black")
                else:
                    pass
        # plot init
        text = ax.text(self.start[0],self.start[1], 'S', ha="center", va="center", color="black")
        # color the cell where the agent is
        im.axes.add_patch(matplotlib.patches.Rectangle((self.start[0]-0.5, self.start[1]-0.5), 1, 1, fill=True, color='green', alpha=0.5)) # type: ignore

        
        plt.show()

    def render_policy(self, goal_idx):
        
        # get coordinates of the goal
        # find coordinates of the goal
        # idx = i*self.Ly+j find i,j
        i = goal_idx//self.Ly
        j = goal_idx%self.Ly
        goal = [i,j]

        fig, ((ax1), (ax2)) = plt.subplots(1,2)

        values = self.values[:,:,goal_idx]
        im1 = ax1.imshow(values, cmap=plt.get_cmap("Spectral")) # type: ignore

        
        optimal_policy_index_QLearning = self.policy[:,:,goal_idx].astype(int)

        # Optimal policy for QLearning as arrows for plots
        optimal_policy_arrows_QLearning = np.zeros( (*self.World.shape, 2) )
        optimal_policy_arrows_QLearning[:,:] = self.actions[ optimal_policy_index_QLearning ]

        # Loop over data dimensions and create text annotations.
        for i in range(self.Lx):
            for j in range(self.Ly):
                if np.logical_and(goal[0]==j, goal[1]==i).any():
                    text = ax1.text(i,j, 'G-10', ha="center", va="center", color="black")
                else:
                    text = ax1.text(i, j, '{:.2f}'.format(values[j, i]), ha="center", va="center", color="black")
        
        
        im2 = ax2.imshow(values, cmap=plt.get_cmap("Spectral")) # type: ignore
        X = np.arange(self.Lx)
        Y = np.arange(self.Ly)

        optimal_policy_arrows_QLearning[goal[0], goal[1]] = 0
            
        U, V = optimal_policy_arrows_QLearning[:,:,1], -optimal_policy_arrows_QLearning[:,:,0]
        q = ax2.quiver(X, Y, U, V, color="black")

        # remove axis ticks
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.show()





class Qlearning_TDControl():
    def __init__(self, space_size, action_size, gamma:float=1, lr_v=0.01):
        """
        Calculates optimal policy using off-policy Temporal Difference control
        Evaluates Q-value for (S,A) pairs, using one-step updates.
        """            
        # the discount factor
        self.gamma = gamma
        # size of system
        self.space_size = space_size # as tuple
        self.action_size = action_size

        # the learning rate
        self.lr_v = lr_v
        
        # where to save returns
        self.Qvalues = np.zeros( (*self.space_size, self.action_size) )
    
    
    def single_step_update(self, s, a, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Uses the BEST (evaluated) action in the new state <- Q(S_new, A*) = max_A Q(S_new, A).
        """
        if done:
            # Q-Learning: deltaQ = R - Q(s,a) 
            deltaQ = r - self.Qvalues[ (*s, a) ]
        else:
            # Q-Learning: deltaQ = R + gamma*max_act Q(new_s, act)- Q(s,a) 
            maxQ_over_actions = np.max( self.Qvalues[ (*new_s,) ] )
            deltaQ = r + self.gamma * maxQ_over_actions - self.Qvalues[ (*s, a) ]

        
        self.Qvalues[ (*s, a) ] += self.lr_v * deltaQ
            
    def get_action_epsilon_greedy(self, s, eps):
        """
        Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        """
        ran = np.random.rand()
                
        if (ran < eps): # if the random number is smaller than epsilon, take a random action
            # by constructing a probability array with equal probabilities for all actions
            prob_actions = np.ones(self.action_size)/self.action_size
        
        else:
            # find the best Qvalue and assign probability 1 (or 1/n, where n is the number of actions which correspond to the best one) 
            # to that action
            best_value = np.max(self.Qvalues[ (*s,) ])
            best_actions = ( self.Qvalues[ (*s,) ] == best_value )
            prob_actions = best_actions / np.sum(best_actions)
            
        # take one action from the array of actions with the probabilities as defined above.
        a = np.random.choice(self.action_size, p=prob_actions)
        return a 
        
    def greedy_policy(self): # when we have finished
        greedy_pol = np.argmax(self.Qvalues, axis=2)
        return greedy_pol
    






