import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import tensorflow_probability as tfp
import minerl
from absl import flags
import argparse
import os

import network

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Minecraft Supervised Learning')

parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--model_name', type=str, help='name of saved model')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')

arguments = parser.parse_args()

if arguments.gpu_use == True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

workspace_path = arguments.workspace_path

num_actions = 20
num_hidden_units = 512

#model = tf.keras.models.load_model('MineRL_SL_Model')

model = network.ActorCritic(num_actions, num_hidden_units)

model.load_weights(workspace_path + "/model/" + arguments.model_name)

# Create the environment
env = gym.make('MineRLTreechop-v0')

seed = 980
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

reward_sum = 0
for i_episode in range(0, 10000):
    observation = env.reset()
    
    inventory_channel = np.zeros((64,64,1))
    if 'inventory' in observation:
        region_max_height = observation['pov'].shape[0]
        region_max_width = observation['pov'].shape[1]
        rs = 8
        if min(region_max_height, region_max_width) < rs:
            raise ValueError("'region_size' is too large.")
            
        num_element_width = region_max_width // rs

        inventory_channel = np.zeros(shape=list(observation['pov'].shape[:-1]) + [1], 
                                     dtype=observation['pov'].dtype)
        #print("state['inventory'].keys(): " + str(state['inventory'].keys()))
        for key_idx, key in enumerate(observation['inventory'].keys()):
            #print("key.shape : " + str(key))
            #print("state['inventory'][key][i] : " + str(state['inventory'][key][i]))
            item_scaled = np.clip(1 - 1 / (observation['inventory'][key] + 1),  # Inversed
                                  0, 1)
            #print("item_scaled : " + str(item_scaled))
            item_channel = np.ones(shape=[rs, rs, 1], dtype=observation['pov'].dtype) * item_scaled
            width_low = (key_idx % num_element_width) * rs
            height_low = (key_idx // num_element_width) * rs

            if height_low + rs > region_max_height:
                raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component.")

            inventory_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel

    state = np.concatenate((observation['pov'] / 255.0, inventory_channel), axis=2)
    state = tf.constant(state, dtype=tf.float32)
    
    memory_state = tf.zeros([1,128], dtype=np.float32)
    carry_state = tf.zeros([1,128], dtype=np.float32)
    step = 0
    while True:
        step += 1

        state = tf.expand_dims(state, 0)
        action_probs, _, memory_state, carry_state = model(state, memory_state, carry_state)
        
        action_dist = tfd.Categorical(probs=action_probs)
        action_index = int(action_dist.sample()[0])
        #print("action_index: ", action_index)
        #if random.random() <= 0.01:
        #    action_index = random.randint(0,18)
        #else:
        #    action_index = np.argmax(np.squeeze(action_probs))
        #print("action_index: ", action_index)
        
        action = env.action_space.noop()
        if (action_index == 0):
            action['camera'] = [0, -5]
            action['attack'] = 0
        elif (action_index == 1):
            action['camera'] = [0, -5]
            action['attack'] = 1
        elif (action_index == 2):
            action['camera'] = [0, 5]
            action['attack'] = 0
        elif (action_index == 3):
            action['camera'] = [0, 5]
            action['attack'] = 1
        elif (action_index == 4):
            action['camera'] = [-5, 0]
            action['attack'] = 0
        elif (action_index == 5):
            action['camera'] = [-5, 0]
            action['attack'] = 1
        elif (action_index == 6):
            action['camera'] = [5, 0]
            action['attack'] = 0
        elif (action_index == 7):
            action['camera'] = [5, 0]
            
        elif (action_index == 8):
            action['forward'] = 1
            action['attack'] = 0
        elif (action_index == 9):
            action['forward'] = 1
            action['attack'] = 1
            
        elif (action_index == 10):
            action['jump'] = 1
            action['attack'] = 0
        elif (action_index == 11):
            action['jump'] = 1
            action['attack'] = 1
            
        elif (action_index == 12):
            action['back'] = 1
            action['attack'] = 0
        elif (action_index == 13):
            action['back'] = 1
            action['attack'] = 1
            
        elif (action_index == 14):
            action['left'] = 1
            action['attack'] = 0
        elif (action_index == 15):
            action['left'] = 1
            action['attack'] = 1
            
        elif (action_index == 16):
            action['right'] = 1
            action['attack'] = 0
        elif (action_index == 17):
            action['right'] = 1
            action['attack'] = 1 
            
        elif (action_index == 18):
            action['sneak'] = 1
            action['attack'] = 0
        elif (action_index == 19):
            action['sneak'] = 1
            action['attack'] = 1 
        
        observation_1, reward, done, info = env.step(action)
        #render(observation_1['pov'])
        
        inventory_channel_1 = np.zeros((64,64,1))
        if 'inventory' in observation_1:
            region_max_height = observation_1['pov'].shape[0]
            region_max_width = observation_1['pov'].shape[1]
            rs = 8
            if min(region_max_height, region_max_width) < rs:
                raise ValueError("'region_size' is too large.")
                
            num_element_width = region_max_width // rs

            inventory_channel_1 = np.zeros(shape=list(observation_1['pov'].shape[:-1]) + [1], 
                                           dtype=observation_1['pov'].dtype)
            #print("state['inventory'].keys(): " + str(state['inventory'].keys()))
            for key_idx, key in enumerate(observation_1['inventory'].keys()):
                #print("key.shape : " + str(key))
                #print("state['inventory'][key][i] : " + str(state['inventory'][key][i]))
                item_scaled = np.clip(1 - 1 / (observation_1['inventory'][key] + 1),  # Inversed
                                      0, 1)
                #print("item_scaled : " + str(item_scaled))
                item_channel = np.ones(shape=[rs, rs, 1], dtype=observation_1['pov'].dtype) * item_scaled
                width_low = (key_idx % num_element_width) * rs
                height_low = (key_idx // num_element_width) * rs

                if height_low + rs > region_max_height:
                    raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component.")

                inventory_channel_1[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel

        next_state = np.concatenate((observation_1['pov'] / 255.0, inventory_channel_1), axis=2)
        next_state = tf.constant(next_state, dtype=tf.float32)
        
        reward_sum += reward

        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            #observation = env.reset()
            break

env.close()
