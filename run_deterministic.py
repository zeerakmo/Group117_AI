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
#Prints out the graphs which show the states values and rewards values agains the time in weeks
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
  #Prints the total reward
  print('total reward', np.sum(rewards))
#Runs the agent in the enviroment by choosing the same specific action every time
def __deterministic_agent(action, env):
  states = []
  rewards = []
  done = False

  s = env.reset()
  states.append(s)
  while not done:
      s, r, done, i = env.step(action) # deterministic agent
      states.append(s)
      rewards.append(r)
  return states, rewards

def deterministic_agent(action, env):
  s,r = __deterministic_agent(action, env)
  #graphs(s,r)
  return s,r
