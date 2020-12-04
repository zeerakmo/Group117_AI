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

#Function for plotting the state values and rewards graphs
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

#Function to create the q table given parameters, initializes table with correct shape, creates lists of bins per state of correct size, iterates through
#episodes and calculating new q values given the formula, updates state and reward
def __qtable(env,num_episodes,learning_rate,discount_rate,exploration_rate,exploration_decay_rate,list_of_vals):
  # num_episodes = 2000

  # learning_rate = 0.01   # notation - η or α 0.2
  # discount_rate = 0.8    #notation - γ(gamma) 0.9

  # exploration_rate = 0.8  #notation - ε
  max_exploration_rate = 1
  min_exploration_rate = 0.1
  # exploration_decay_rate = 0.01

  
  # q_table = np.zeros((100,100,100,100,4),dtype=float)
  global q_table
  q_table = np.zeros((list_of_vals[0],list_of_vals[1],list_of_vals[2],list_of_vals[3],4),dtype=float)

  import math
  global bins_s0
  bins_s0 = []
  global bins_s1
  bins_s1=[]
  global bins_s2
  bins_s2=[]
  global bins_s3
  bins_s3=[]
  y=0

  for x in range(list_of_vals[0]):
      y+=math.ceil(600000000/list_of_vals[0])
      bins_s0.append(y)

  y=0
  for x in range(list_of_vals[1]):
      y+=math.ceil(600000000/list_of_vals[1])
      bins_s1.append(y)

  y=0
  for x in range(list_of_vals[2]):
      y+=math.ceil(600000000/list_of_vals[2])
      bins_s2.append(y)

  y=0
  for x in range(list_of_vals[3]):
      y+=math.ceil(600000000/list_of_vals[3])
      bins_s3.append(y)

  rewards_all_episodes = []  
  episode_steps = []
  states= []

  for episode in range(num_episodes):
      state = env.reset()
      
      done = False
      rewards_current_episode = 0

      while not done:
        exploration_rate_threshold = np.random.uniform(0, 1)   
        if exploration_rate_threshold > exploration_rate:
          s_indice_0 = np.digitize(state[0],bins_s0)
          s_indice_1 = np.digitize(state[1],bins_s1)
          s_indice_2 = np.digitize(state[2],bins_s2)
          s_indice_3 = np.digitize(state[3],bins_s3)
          
          action = np.argmax(q_table[s_indice_0,s_indice_1,s_indice_2,s_indice_3, :]) 
          
        else:
          action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)  #tuple unpacking

        indice_0 = np.digitize(state[0],bins_s0)
        indice_1 = np.digitize(state[1],bins_s1)
        indice_2 = np.digitize(state[2],bins_s2)
        indice_3 = np.digitize(state[3],bins_s3)

        ns_indice_0 = np.digitize(new_state[0],bins_s0)
        ns_indice_1 = np.digitize(new_state[1],bins_s1)
        ns_indice_2 = np.digitize(new_state[2],bins_s2)
        ns_indice_3 = np.digitize(new_state[3],bins_s3)

    
        q_table[indice_0,indice_1,indice_2,indice_3, action] = q_table[indice_0,indice_1,indice_2,indice_3, action] + learning_rate * (reward + discount_rate * (np.max(q_table[ns_indice_0,ns_indice_1,ns_indice_2,ns_indice_3, :]))- q_table[indice_0,indice_1,indice_2,indice_3, action])

        state = new_state               
        rewards_current_episode += reward

      exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
      rewards_all_episodes.append(rewards_current_episode)
  return q_table,rewards_all_episodes    


#Function is a caller function, takes parameters and calls Q table constructor function as well as the graph to plot the convergence of reward
def qtable(env,num_episodes = 100,learning_rate = 0.01,discount_rate = 0.8,exploration_rate = 0.8,exploration_decay_rate = 0.01,list_of_vals=[100,50,50,100]):
  # num_episodes = 100,learning_rate = 0.01,discount_rate = 0.8,exploration_rate = 0.8,exploration_decay_rate = 0.01,list_of_items=[100,50,50,100] 
  q_table,total_r = __qtable(env,num_episodes,learning_rate,discount_rate,exploration_rate,exploration_decay_rate,list_of_vals)
  graphCon(total_r)
  return q_table

  # s,r=run(q_table,env)
  # graphs(s,r)

#Function called to actually run the agent and iterate through the Q table given the returned action
def run(q_table, env):
  states = []
  rewards = []
  done = False

  state = env.reset()
  states.append(state)
  while not done:
    
      s_indice_0 = np.digitize(state[0],bins_s0)
      s_indice_1 = np.digitize(state[1],bins_s1)
      s_indice_2 = np.digitize(state[2],bins_s2)
      s_indice_3 = np.digitize(state[3],bins_s3)

      action = np.argmax(q_table[s_indice_0,s_indice_1,s_indice_2,s_indice_3, :])      
      state, r, done, i = env.step(action)
      # print(q_table[s_indice_0,s_indice_1,s_indice_2,s_indice_3, :])
      states.append(state)
      rewards.append(r)
  q_table =None
  del q_table
  return states,rewards
 
#Function for plotting the convergence graph of episode reward over time 
def graphCon(rewards):
  import pandas as pd
  fig2 = plt.figure(figsize=(10,5))
  smoothing_window = 500
  rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
