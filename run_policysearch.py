import gym

import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np

import virl
env = virl.Epidemic(stochastic=False, noisy=False)
import matplotlib.pyplot as plt
import time
import itertools
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

import numpy as np
import sys
import os
import random
from collections import namedtuple
import collections
import copy

# Import the open AI gym
import gym

# %tensorflow_version 1.x
import tensorflow as tf
import keras
from keras.models import Sequential
#from keras.layers import Dense
from keras.optimizers import Adam
#from keras import backend as K

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import initializers



from gym.envs.toy_text import discrete



import pandas as pd
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
#Graphs the reward and states against each week
def graphs(states,rewards):  
  fig, axes = plt.subplots(1, 2, figsize=(20, 8))
  labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
  states = np.array(states)
  for i in range(4):
      axes[0].plot(states[:,i], label=labels[i]);
  axes[0].set_xlabel('weeks since start of epidemic')
  axes[0].set_ylabel('State s(t)')
  axes[0].legend()
  axes[1].plot(rewards);
  axes[1].set_title('Reward')
  axes[1].set_xlabel('weeks since start of epidemic')
  axes[1].set_ylabel('reward r(t)')

  print('total reward', np.sum(rewards))
  
def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


class NeuralNetworkPolicyEstimator():
    """ 
    A very basic MLP neural network approximator and estimator for poliy search    
    
    The only tricky thing is the traning/loss function and the specific neural network arch
    """
    
    def __init__(self, alpha, n_actions, d_states, nn_config, verbose=False):        
        self.alpha = alpha 
        self.nn_config = nn_config   
        self.n_actions = n_actions        
        self.d_states = d_states
        self.verbose=verbose # Print debug information        
        self.n_layers = len(nn_config)  # number of hidden layers        
        self.model = []
        self.__build_network(d_states, n_actions)
        self.__build_train_fn()
             

    def __build_network(self, input_dim, output_dim):
        """Create a base network usig the Keras functional API"""
        self.X = layers.Input(shape=(input_dim,))
        net = self.X
        for h_dim in nn_config:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)
        net = layers.Dense(output_dim, kernel_initializer=initializers.Zeros())(net)
        net = layers.Activation("softmax")(net)
        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        """ Create a custom train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.        
        Called using self.train_fn([state, action_one_hot, target])`
        which would train the model. 
        
        Hint: you can think of K. as np.
        
        """
        # predefine a few variables
        action_onehot_placeholder   = K.placeholder(shape=(None, self.n_actions),name="action_onehot") # define a variable
        target                      = K.placeholder(shape=(None,), name="target") # define a variable     
        # this part defines the loss and is very important!
        action_prob        = self.model.output # the outlout of the neural network     
        action_selected_prob        = K.sum(action_prob * action_onehot_placeholder, axis=1) # probability of the selcted action      
        log_action_prob             = K.log(action_selected_prob) # take the log
        loss = -log_action_prob * target # the loss we are trying to minimise
        loss = K.mean(loss)
        # defining the speific optimizer to use
        adam = optimizers.Adam(lr=self.alpha)# clipnorm=1000.0) # let's use a kereas optimiser called Adam
        updates = adam.get_updates(params=self.model.trainable_weights,loss=loss) # what gradient updates to we parse to Adam            
        # create a handle to the optimiser function    
        self.train_fn = K.function(inputs=[self.model.input,action_onehot_placeholder,target],
                                   outputs=[],
                                   updates=updates) # return a function which, when called, takes a gradient step    
      
    
    def predict(self, s, a=None):              
        if a==None:            
            return self._predict_nn(s)
        else:                        
            return self._predict_nn(s)[a]
        
    def _predict_nn(self,state_hat):                          
        """
        Implements a basic MLP with tanh activations except for the final layer (linear)               
        """     
        x = self.model.predict(state_hat)                                                   
        return x
  
    def update(self, states, actions, target):  
        """
            states: a interger number repsenting the discrete state
            actions: a interger number repsenting the discrete action
            target: a real number representing the discount furture reward, reward to go
            
        """
        action_onehot = np_utils.to_categorical(actions, num_classes=self.n_actions) # encodes the state as one-hot
        self.train_fn([states, action_onehot, target]) # call the custom optimiser which takes a gradient step
        return 
        
    def new_episode(self):        
        self.t_episode  = 0. 

def reinforce(env, estimator_policy, num_episodes, state_space, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized         
        num_episodes: Number of episodes to run for
        discount_factor: reward discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    
    Adapted from: https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb
    """

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    states = []
    rewards = []

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        states.append(state)
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            #Place the states in the corresponding bins
            # Take a step
            s_indice_0 = np.digitize(state[0],s0bins)
            s_indice_1 = np.digitize(state[1],s1bins)
            s_indice_2 = np.digitize(state[2],s2bins)
            s_indice_3 = np.digitize(state[3],s3bins)
            #Pad the values with a 0 to refer to them correctly
            if len(str(s_indice_0))==1:
              x0 = "0"+str(s_indice_0)
            if len(str(s_indice_1))==1:
              x1 = "0"+str(s_indice_1)
            if len(str(s_indice_2))==1:
              x2 = "0"+str(s_indice_2)
            if len(str(s_indice_3))==1:
              x3 = "0"+str(s_indice_3)

            #Create the string representation of the bin indices
            s = x0+x1+x2+x3
            #One hot enocde the values for the specific unique one hot encoding
            value = oh_dict[s]
            state_oh = np.zeros((1,state_space))
            state_oh[0,value] = 1.0 
                               
            action_probs = estimator_policy.predict(state_oh)
            action_probs = action_probs.squeeze()
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)
            states.append(next_state)
            rewards.append(reward)
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")            

            if done:
                break
                
            state = next_state
            # print(state)
    
        # Go through the episode, step-by-step and make policy updates (note we sometime use j for the individual steps)
        estimator_policy.new_episode()
        new_theta=[]
        for t, transition in enumerate(episode):     
            #Place the state values in the correct bins and pad the value with an extra 0 to ensure its correctly represented

            s_indice_0 = np.digitize(transition.state[0],s0bins)
            s_indice_1 = np.digitize(transition.state[1],s1bins)
            s_indice_2 = np.digitize(transition.state[2],s2bins)
            s_indice_3 = np.digitize(transition.state[3],s3bins)

            if len(str(s_indice_0))==1:
              x0 = "0"+str(s_indice_0)
            if len(str(s_indice_1))==1:
              x1 = "0"+str(s_indice_1)
            if len(str(s_indice_2))==1:
              x2 = "0"+str(s_indice_2)
            if len(str(s_indice_3))==1:
              x3 = "0"+str(s_indice_3)

            
            s = x0+x1+x2+x3
    
            #One hot encode the unique state representation 
            value = oh_dict[s]
            state_oh = np.zeros((1,state_space))
            state_oh[0,value] = 1.0 


            # for i,x in enumerate(b):
            #   if np.array_equal(x,s):
            #     # print(i)
            #     state_oh = np.zeros((1,1296))
            #     state_oh[0,i] = 1.0 
            
            # The return, G_t, after this timestep; this is the target for the PolicyEstimator
            G_t = sum(discount_factor**i * t.reward for i, t in enumerate(episode[t:]))
            # Update our policy estimator
            estimator_policy.update(state_oh, transition.action,np.array(G_t))
         
    return stats, states, rewards


def runpolicy(env):
  states = []
  rewards = []
  done = False

  state = env.reset()
  states.append(state)
  while not done:
      # s_indice_0 = np.digitize(s[0],bins)           
      # state_oh = np.zeros((1,6))
      # state_oh[0,s_indice_0] = 1.0  

      s_indice_0 = np.digitize(state[0],s0bins)
      s_indice_1 = np.digitize(state[1],s1bins)
      s_indice_2 = np.digitize(state[2],s2bins)
      s_indice_3 = np.digitize(state[3],s3bins)
   

      if len(str(s_indice_0))==1:
        x0 = "0"+str(s_indice_0)
      if len(str(s_indice_1))==1:
        x1 = "0"+str(s_indice_1)
      if len(str(s_indice_2))==1:
        x2 = "0"+str(s_indice_2)
      if len(str(s_indice_3))==1:
        x3 = "0"+str(s_indice_3)


      s = x0+x1+x2+x3
      value = oh_dict[s]
      state_oh = np.zeros((1,n_states))
      state_oh[0,value] = 1.0
      
      
      choices = policy_estimator.predict(state_oh)
      action_probs = choices.squeeze()

      #Choose an action depending on the probabilties for eaceh action for what state we are in
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

      state, r, done, i = env.step(action) # deterministic agent
      states.append(state)
      rewards.append(r)
  return states,rewards


def policy_execute(env, alpha = 0.001, discount_factor=0.99, episodes_number = 500, list_of_vals = [6,50,50,6],test_env=None):
    import math
    global s0bins
    s0bins = []
    global s1bins
    s1bins=[]
    global s2bins
    s2bins=[]
    global s3bins
    s3bins=[]
    #Create the bins with the specified interval values from the parameters 
    y=0
    for x in range(list_of_vals[0]):
        y+=math.ceil(600000000/list_of_vals[0])
        s0bins.append(y)

    y=0
    for x in range(list_of_vals[1]):
        y+=math.ceil(600000000/list_of_vals[1])
        s1bins.append(y)

    y=0
    for x in range(list_of_vals[2]):
        y+=math.ceil(600000000/list_of_vals[2])
        s2bins.append(y)

    y=0
    for x in range(list_of_vals[3]):
        y+=math.ceil(600000000/list_of_vals[3])
        s3bins.append(y)

    if test_env==None:
      test_env=env
    #Create the array of unique state combinations
    global a 
    a = np.mgrid[0:list_of_vals[0], 0:list_of_vals[1], 0:list_of_vals[2], 0:list_of_vals[3]]     # All points in a 3D grid within the given ranges
    # a = a.reshape([1296,4])
    a = np.rollaxis(a, 0, 5)         # Make the 0th axis into the last axis
    a = a.reshape((list_of_vals[0]*list_of_vals[1]*list_of_vals[2]*list_of_vals[3] , 4))
    global b
    b = np.unique(a, axis=1)
    #Implement the unique states as the keys in a dictionary whos values will point to their one hot encodings
    global oh_dict
    oh_dict={}
    for i,x in enumerate(b):
        x0=str(x[0])
        x1=str(x[1])
        x2=str(x[2])
        x3=str(x[3])

        if len(str(x[0]))==1:
            x0 = "0"+str(x[0])
        if len(str(x[1]))==1:
            x1 = "0"+str(x[1])
        if len(str(x[2]))==1:
            x2 = "0"+str(x[2])
        if len(str(x[3]))==1:
            x3 = "0"+str(x[3])
        global combination 
        combination = x0+x1+x2+x3

        oh_dict[combination]= i

    from keras.utils import to_categorical
    # Instantiate a PolicyEstimator (i.e. the function-based approximation)
   
    global n_actions 
    n_actions = 4
    global n_states
    n_states = list_of_vals[0]*list_of_vals[1]*list_of_vals[2]*list_of_vals[3]
    global nn_config  
    nn_config = [] # number of neurons in the hidden layers, should be [] as default
    global policy_estimator
    policy_estimator = NeuralNetworkPolicyEstimator(alpha, n_actions, n_states, nn_config)
    stats, s, r = reinforce(env, policy_estimator, episodes_number, n_states, discount_factor)

    plot_episode_stats(stats, smoothing_window=25)
