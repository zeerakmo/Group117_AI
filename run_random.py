import gym

import os
os.chdir('..')
from matplotlib import pyplot as plt
import numpy as np

# import virl
# env = virl.Epidemic(stochastic=False, noisy=False)
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

#The following takes in the states and rewards for the agent and graphs the results
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

#Returns the usable values for state and action
def random_agent(iterations, env):
  s,r = avg_random(iterations, env)
  #graphs(s,r)
  return s, r


#The random agent makes random decisions at every given opportunity 
def __random_agent(env):
  states = []
  rewards = []
  done = False

  s = env.reset()
  states.append(s)
  while not done:
      s, r, done, i = env.step(action=np.random.choice(env.action_space.n))
      states.append(s)
      rewards.append(r)
  return states, rewards


#Averages the results of the random agents against the number of specified iterations
def avg_random(iterations, env):
  cumulative_s = []
  cumulative_r = []
  total_s=np.zeros(53*4)
  total_s.shape = (53,4)

  total_r=np.zeros(52)
  total_r.shape = (52)

  for i in range(iterations):
    s, r = __random_agent(env)
    cumulative_s.append(s)
    cumulative_r.append(r)

  for i in range(iterations):
    total_s = np.add(total_s,cumulative_s[i])
    total_r = np.add(total_r,cumulative_r[i])
 
  res_s = np.array(total_s)/iterations
  res_r = np.array(total_r)/iterations
  return res_s,res_r
