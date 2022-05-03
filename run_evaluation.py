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
from utils import TrajectoryInformation, DummyDataLoader, TrajectoryDataPipeline, create_nodes

tfd = tfp.distributions

parser = argparse.ArgumentParser(description='Minecraft Evaluation')

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

num_actions = 43
num_hidden_units = 512

#model = tf.keras.models.load_model('MineRL_SL_Model')

model = network.ActorCritic(num_actions, num_hidden_units)

#model.load_weights(workspace_path + "/model/" + arguments.model_name)
model.load_weights("model_tree/" + arguments.model_name)

# Create the environment
#env = gym.make('MineRLObtainDiamondDense-v0')
env = gym.make('MineRLTreechop-v0')

seed = 980
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

reward_sum = 0

data_path = '/home/kimbring2/minerl_data/'
env_name = 'MineRLObtainDiamondDense-v0'
trajectory_name = 'v3_absolute_grape_changeling-6_37339-46767'

item_list = ['log', 'planks', 'crafting_table', 'stick', 'wooden_pickaxe', 'cobblestone', 'coal', 
             'stone_pickaxe', 'iron_ore', 'furnace', 'iron_ingot', 'iron_pickaxe', 'stone']

node_index = 0

trajectory_information = TrajectoryInformation(data_path + env_name + "/" + trajectory_name)
final_chain = trajectory_information.to_old_chain_format(items=trajectory_information.chain, return_time_indexes=False)

nodes = create_nodes(final_chain)
for node in nodes:
    print("node.name: ", node.name)

    
for i_episode in range(0, 10000):
    observation = env.reset()
    
    item_index = item_list.index(nodes[node_index].name)
    scale = len(item_list)
    item_layer = np.zeros([64,64,scale], dtype=np.float32)
    item_layer[:, :, item_index] = 1.0
    
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
    state = np.concatenate((state, item_layer), axis=2)
    state = tf.constant(state, dtype=tf.float32)

    state = observation['pov'] / 255.0
    
    memory_state = tf.zeros([1,128], dtype=np.float32)
    carry_state = tf.zeros([1,128], dtype=np.float32)
    step = 0
    while True:
        step += 1

        state = tf.expand_dims(state, 0)
        #print("state.shape: ", state.shape)
        action_probs, _, memory_state, carry_state = model(state, memory_state, carry_state)
        
        action_dist = tfd.Categorical(logits=action_probs)
        if random.random() <= 0.05:
            action_index = random.randint(0,42)
        else:
            action_index = int(action_dist.sample()[0])
            
        #print("action_index: ", action_index)
        
        action = env.action_space.noop()
        if (action_index == 0):
            action['camera'] = [0, -2.5]
            action['attack'] = 1
        elif (action_index == 1):
            action['camera'] = [0, -2.5]
            action['forward'] = 1
        elif (action_index == 2):
            action['camera'] = [0, -2.5]
            action['left'] = 1
        elif (action_index == 3):
            action['camera'] = [0, -2.5]
            action['right'] = 1
        elif (action_index == 4):
            action['camera'] = [0, -2.5]
            action['back'] = 1
        elif (action_index == 5):
            action['camera'] = [0, -2.5]
            action['jump'] = 1
        elif (action_index == 6):
            action['camera'] = [0, -2.5]
            
        elif (action_index == 7):
            action['camera'] = [0, 2.5]
            action['attack'] = 1
        elif (action_index == 8):
            action['camera'] = [0, 2.5]
            action['forward'] = 1
        elif (action_index == 9):
            action['camera'] = [0, 2.5]
            action['left'] = 1
        elif (action_index == 10):
            action['camera'] = [0, 2.5]
            action['right'] = 1
        elif (action_index == 11):
            action['camera'] = [0, 2.5]
            action['back'] = 1
        elif (action_index == 12):
            action['camera'] = [0, 2.5]
            action['jump'] = 1
        elif (action_index == 13):
            action['camera'] = [0, 2.5]
            
        elif (action_index == 14):
            action['camera'] = [-2.5, 0]
            action['attack'] = 1
        elif (action_index == 15):
            action['camera'] = [-2.5, 0]
            action['forward'] = 1
        elif (action_index == 16):
            action['camera'] = [-2.5, 0]
            action['left'] = 1
        elif (action_index == 17):
            action['camera'] = [-2.5, 0]
            action['right'] = 1
        elif (action_index == 18):
            action['camera'] = [-2.5, 0]
            action['back'] = 1
        elif (action_index == 19):
            action['camera'] = [-2.5, 0]
            action['jump'] = 1
        elif (action_index == 20):
            action['camera'] = [-2.5, 0]
            
        elif (action_index == 21):
            action['camera'] = [2.5, 0]
            action['attack'] = 1
        elif (action_index == 22):
            action['camera'] = [2.5, 0]
            action['forward'] = 1
        elif (action_index == 23):
            action['camera'] = [2.5, 0]
            action['left'] = 1
        elif (action_index == 24):
            action['camera'] = [2.5, 0]
            action['right'] = 1
        elif (action_index == 25):
            action['camera'] = [2.5, 0]
            action['back'] = 1
        elif (action_index == 26):
            action['camera'] = [2.5, 0]
            action['jump'] = 1
        elif (action_index == 27):
            action['camera'] = [2.5, 0]
            
        elif (action_index == 28):
            action['forward'] = 1
            action['attack'] = 1
        elif (action_index == 29):
            action['forward'] = 1
            action['jump'] = 1
        elif (action_index == 30):
            action['forward'] = 1
            
        elif (action_index == 31):
            action['jump'] = 1
        elif (action_index == 32):
            action['jump'] = 1
            action['attack'] = 1
            
        elif (action_index == 33):
            action['back'] = 1
            action['attack'] = 1
        elif (action_index == 34):
            action['back'] = 1
            
        elif (action_index == 35):
            action['left'] = 1
            action['attack'] = 1
        elif (action_index == 36):
            action['left'] = 1
            
        elif (action_index == 37):
            action['right'] = 1
            action['attack'] = 1 
        elif (action_index == 38):
            action['right'] = 1
            
        elif (action_index == 39):
            action['sneak'] = 1
            action['attack'] = 1 
        elif (action_index == 40):
            action['sneak'] = 1
            
        elif (action_index == 41):
            action['attack'] = 1 
        
        #craft_action = nodes[node_index].crafting_agent.get_crafting_action()
        #print("craft_action: ", craft_action)
        
        #action = {**action, **craft_action}
        #print("action: ", action)
        
        observation_1, reward, done, info = env.step(action)
        #print("observation_1['inventory'][nodes[node_index].name]: ", observation_1['inventory'][nodes[node_index].name])
        
        #print("")
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

        item_index = item_list.index(nodes[node_index].name)
        scale = len(item_list)
        item_layer = np.zeros([64,64,scale], dtype=np.float32)
        item_layer[:, :, item_index] = 1.0
                
        next_state = np.concatenate((observation_1['pov'] / 255.0, inventory_channel_1), axis=2)
        next_state = np.concatenate((next_state, item_layer), axis=2)
        next_state = tf.constant(next_state, dtype=tf.float32)
        next_state = observation_1['pov'] / 255.0

        reward_sum += reward

        state = next_state
        if done:
            print("Total reward: {:.2f},  Total step: {:.2f}".format(reward_sum, step))
            step = 0
            reward_sum = 0  
            #observation = env.reset()
            break

env.close()
