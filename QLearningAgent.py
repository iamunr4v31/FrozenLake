import numpy as np
import pickle

class Agent:
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_decay, Q=None):
        """Initialize the agent

        Args:
            lr (float): learning rate
            gamma (float): discount factor
            n_actions (int): action size
            n_states (int): state size
            eps_start (float): initial epsilon
            eps_end (float): target epsilon
            eps_decay (float): decay rate for epsilon
        """        
        self.lr =  lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay

        self.Q = {}
        self.init_Q(Q)

    def init_Q(self, Q):
        """
        Initialize the Q table (n_states x n_actions) with zeros
        """
        if Q is None:
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    self.Q[(state, action)] = 0.0           # create a dictionary of Q values for each state and action
        else:
            self.Q = Q
    
    def choose_action(self, state):
        """epsilon-greedy action selection

        Args:
            state (int): current state

        Returns:
            index: index of the action to be taken
        """
        if np.random.random() < self.epsilon:
            return np.random.choice([i for i in range(self.n_actions)])     # choose a random action
        else:
            return np.array([self.Q[state, a] for a in range(self.n_actions)]).argmax()  # choose the best action
    
    def decrement_epsilon(self):
        """epsilon decay"""
        self.epsilon = self.epsilon * self.eps_decay if self.epsilon > self.eps_min else self.eps_min   # decay epsilon
    
    def learn(self, state, action, reward, next_state):
        """Step update of Q-value

        Args:
            state (int): current state
            action (int): action chosen
            reward (int): reward corresponding to state and action
            next_state (int): next state corresponding to state and action
        """ 
        actions = np.array([self.Q[next_state, a] for a in range(self.n_actions)])  # get the Q values for the next state
        a_max = np.argmax(actions)      # get the index of the best action or maximal action for next state
        self.Q[state, action] += self.lr * (reward + self.gamma * self.Q[next_state, a_max] - self.Q[state, action])  # update the Q value for the current state and action
        self.decrement_epsilon()
    
    def save(self, filename):
        """Save the agent parameters"""        
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        """Load the agent parameters"""
        with open(filename, "rb") as f:
            return pickle.load(f)
