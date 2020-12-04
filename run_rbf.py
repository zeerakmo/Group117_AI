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


# %matplotlib inline
from collections import namedtuple

import pandas as pd
# Let's import basic tools for defining the function and doing the gradient-based learning
import sklearn.pipeline
import sklearn.preprocessing
#from sklearn.preprocessing import PolynomialFeatures # you can try with polynomial basis if you want (It is difficult!)
from sklearn.linear_model import SGDRegressor # this defines the SGD function
from sklearn.kernel_approximation import RBFSampler # this is the RBF function transformation method

observation_examples = np.array([env.observation_space.sample() for x in range(3000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

feature_transformer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=1000)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=1000)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=1000)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=1000))
        ])
feature_transformer.fit(scaler.transform(observation_examples))


class FunctionApproximator():
    """
    Q(s,a) function approximator. 

    it uses a specific form for Q(s,a) where seperate functions are fitteted for each 
    action (i.e. four Q_a(s) individual functions)

    We could have concatenated the feature maps with the action TODO TASK?

    """
 
    def __init__(self, eta0= 0.001, learning_rate= "constant"):
        #
        # Args:
        #   eta0: learning rate (initial), default 0.01
        #   learning_rate: the rule used to control the learning rate;
        #   see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html for details
        #        
        # We create a seperate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up and understand.
        #
        #
        self.eta0=eta0
        self.learning_rate=learning_rate
        
        self.models = []
        for _ in range(env.action_space.n):

            # You may want to inspect the SGDRegressor to fully understand what is going on
            # ... there are several interesting parameters you may want to change/tune.
            model = SGDRegressor(learning_rate=learning_rate, tol=1e-5, max_iter=1e5, eta0=eta0)
            
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        s_scaled = scaler.transform([state])
        s_transformed = feature_transformer.transform(s_scaled)
        return s_transformed[0]
    
    def predict(self, s, a=None):
        """
        Makes Q(s,a) function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if a==None:
            return np.array([m.predict([features])[0] for m in self.models])
        else:            
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, td_target):
        """
        Updates the approximator's parameters (i.e. the weights) for a given state and action towards
        the target y (which is the TD target).
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [td_target]) # recall that we have a seperate funciton for each a 


# Keep track of some stats
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

# Main Q-learner
def q_learning(env, func_approximator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=0.999):
    """
    Q-Learning algorithm for Q-learning using Function Approximations.
    Finds the optimal greedy policy while following an explorative greedy policy.
    
    Args:
        env: OpenAI environment.
        func_approximator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Exploration strategy; chance the sample a random action. Float between 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    for i_episode in range(num_episodes):
        
        # Create a handle to the policy we're following (including exploration)a
        policy = create_policy(
            func_approximator, epsilon * epsilon_decay**i_episode, env.action_space.n)
        
        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()
        
        # Reset the environment and pick the first action
        state = env.reset()
                 
        # One step in the environment
        for t in itertools.count():
                        
            # Choose an action to take                        
            action_probs, q_vals = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            
            # Take a step
            next_state, reward, done, _ = env.step(action)
    
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            q_values_next = func_approximator.predict(next_state)
            
            # Q-Value TD Target
            td_target = reward + discount_factor * np.max(q_values_next)
              
            # Update the function approximator using our target
            func_approximator.update(state, action, td_target)
                
            if done:
                break
                
            state = next_state


    return stats


def create_policy(func_approximator, epsilon, nA):
    """
    Creates an greedy policy with the exploration defined by the epsilon and nA parameters
    
    Input:
        func_approximator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Output:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(state):
        """
        Input:
            state: a 2D array with the position and velocity
        Output:
            A,q_values: 
        """
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = func_approximator.predict(state)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A,q_values  # return the potentially stochastic policy (which is due to the exploration)

    return policy_fn # return a handle to the function so we can call it in the future


def exec_policy(env, func_approximator, verbose=False):
    """
        A function for executing a policy given the funciton
        approximation (the exploration is zero)
    """
    states=[]
    rewards = []
    actions = []
    # The policy is defined by our function approximator (of the utility)... let's get a hdnle to that function
    policy = create_policy(func_approximator, 0.0, env.action_space.n)
            
    # Reset the environment and pick the first action
    state = env.reset()
    states.append(state)     
    # One step in the environment
    for t in itertools.count():
        #env.render()

        # The policy is stochastic due to exploration 
        # i.e. the policy recommends not only one action but defines a 
        # distrbution , \pi(a|s)
        pi_action_state, q_values = policy(state)
        action = np.random.choice(np.arange(len(pi_action_state)), p=pi_action_state)
        #print("Action (%s): %s" % (action_probs,action)

        # Execute action and observe where we end up incl reward
        next_state, reward, done, _ = env.step(action)
        states.append(next_state)
        rewards.append(reward)
        actions.append(action)
        
        if verbose:
            print("Step %d/51:\n" % (t), end="")
            print("\t state     : %s\n" % (state), end="")            
            print("\t q_approx  : %s\n" % (q_values.T), end="")
            print("\t pi(a|s)   : %s\n" % (pi_action_state), end="")            
            print("\t action    : %s\n" % (action), end="")
            print("\t next_state: %s\n" % (next_state), end="")
            print("\t reward    : %s\n" % (reward), end="")                        
        else:
            print("\rStep {}".format(t), end="")

        
        if done:
            return (states,rewards,actions)
            break

            
        state = next_state


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


def rbf_execute(env, eta0=0.05, learning_rate="constant", num_episodes = 500, epsilon = 1, epsilon_decay=0.7):
  global my_func_approximator
  my_func_approximator = FunctionApproximator(eta0, learning_rate) #was 0.0001, optimal 0.00000001

  stats = q_learning(env, my_func_approximator, num_episodes, epsilon, epsilon_decay) #was 0.6, 1.0

  # states,rewards,actions = exec_policy(env, my_func_approximator,verbose = True)

  # graphs(states,rewards)

  plot_episode_stats(stats, smoothing_window=25)


def run_rbfpolicy(env):
  states,rewards,actions = exec_policy(env, my_func_approximator,verbose = False)
  return states,rewards
