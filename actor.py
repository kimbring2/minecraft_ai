import zmq
import tensorflow as tf
import numpy as np
import collections
import cv2
import gym
import minerl
import cv2
import matplotlib.pyplot as plt

import argparse
from absl import flags
from absl import logging

import os
from utils import TrajectoryInformation, DummyDataLoader, TrajectoryDataPipeline, create_nodes

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='MineRL IMPALA Actor')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
arguments = parser.parse_args()

writer = tf.summary.create_file_writer("tensorboard")

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + arguments.env_id))

data_path = '/home/kimbring2/minerl_data/'
env_name = 'MineRLObtainDiamondDense-v0'
trajectory_name = 'v3_absolute_grape_changeling-6_37339-46767'

trajectory_information = TrajectoryInformation(data_path + env_name + "/" + trajectory_name)
final_chain = trajectory_information.to_old_chain_format(items=trajectory_information.chain, return_time_indexes=False)
nodes = create_nodes(final_chain)
#for node in nodes:
#    print("node.name: ", node.name)

item_list = ['log', 'planks', 'crafting_table', 'stick', 'wooden_pickaxe', 'cobblestone', 'coal', 
             'stone_pickaxe', 'iron_ore', 'furnace', 'iron_ingot', 'iron_pickaxe', 'stone']

#env = gym.make('MineRLNavigateDense-v0')
#env = gym.make('MineRLObtainDiamondDense-v0')
env = gym.make('MineRLTreechop-v0')

num_actions = 43
state_size = (64,64,3)

scores = []
episodes = []
average = []
for episode_step in range(0, 2000000):
    observation = env.reset()
    
    node_index = 0
    
    if nodes[node_index].name in item_list:
        item_index = item_list.index(nodes[node_index].name)
    else:
        item_index = 0

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
        for key_idx, key in enumerate(observation['inventory'].keys()):
            item_scaled = np.clip(1 - 1 / (observation['inventory'][key] + 1),  # Inversed
                                  0, 1)
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

    '''
    pov_array = observation['pov'] / 255.0
    compassAngle_array = observation['compassAngle'] / 360.0
    compassAngle_array = np.ones((64,64,1)) * compassAngle_array

    state = np.concatenate((pov_array, compassAngle_array), 2)
    state = np.expand_dims(state, 0)
    '''

    done = False
    reward = 0.0
    reward_sum = 0
    step = 0
    while True:
        try:
            state_reshaped = np.reshape(state, (1,*state_size)) 

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward / 10.0,
                          "done": done, 
                          "observation": state_reshaped}
            socket.send_pyobj(env_output)
            action_index = int(socket.recv_pyobj()['action'])
            print("action_index:" , action_index)
            
            action = env.action_space.noop()

            '''
            if (action_index == 0):
                action['camera'] = [0, -2.5]
            elif (action_index == 1):
                action['camera'] = [0, 2.5]
            elif (action_index == 2):
                action['forward'] = 1
                action['jump'] = 1
            '''

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
            #action['place'] = 'none'
            #print("node_index: ", node_index)
            #print("action: ", action)
            #if node_index == 2:
            #    action['craft'] = 0
            #    if action['craft'] == 3:
            #        action['craft'] = 'none'
            #
            #if action['craft'] != 'none':
            #    print("action['craft']: ", action['craft'])
            #print("")

            observation_1, reward, done, info = env.step(action)
            
            #reward += -0.005
            
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

            if nodes[node_index].name in item_list:
                item_index = item_list.index(nodes[node_index].name)
            else:
                item_index = 0

            scale = len(item_list)
            item_layer = np.zeros([64,64,scale], dtype=np.float32)
            item_layer[:, :, item_index] = 1.0
            
            next_state = np.concatenate((observation_1['pov'] / 255.0, inventory_channel_1), axis=2)
            next_state = np.concatenate((next_state, item_layer), axis=2)
            next_state = tf.constant(next_state, dtype=tf.float32)
            
            next_state = observation_1['pov'] / 255.0
            
            '''
            #print("node_index: ", node_index)
            next_pov_array = observation_1['pov'] / 255.0
            next_compassAngle_array = observation_1['compassAngle'] / 360.0
            next_compassAngle_array = np.ones((64,64,1)) * next_compassAngle_array

            next_state = np.concatenate((next_pov_array, next_compassAngle_array), 2)
            next_state = np.expand_dims(next_state, 0)
            '''

            state = next_state
            observation_ = observation_1
            reward_sum += reward
            step += 1
            
            '''
            if nodes[node_index].name not in item_list:
                node_index += 1
            elif observation_1['inventory'][nodes[node_index].name] != 0:
                log_num = observation_1['inventory']['log']
                planks_num = observation_1['inventory']['planks']
                crafting_table_num = observation_1['inventory']['crafting_table']
                stick_num = observation_1['inventory']['stick']
                
                print("log_num: ", log_num)
                print("planks_num: ", planks_num)
                print("crafting_table_num: ", crafting_table_num)
                print("stick_num: ", stick_num)
                
                if node_index == 0:
                    if observation_1['inventory'][nodes[node_index].name] >= 4:
                        node_index += 1 
                else:
                    node_index += 1
                    
                print("node_index: ", node_index)
                print("nodes[node_index].name: ", nodes[node_index].name)
                print("")
            '''
            if done:
                if arguments.env_id == 0:
                    scores.append(reward_sum)
                    episodes.append(episode_step)
                    average.append(sum(scores[-50:]) / len(scores[-50:]))

                    with writer.as_default():
                        tf.summary.scalar("average_reward", average[-1], step=episode_step)
                        writer.flush()

                    print("average_reward: " + str(average[-1]))
                else:
                    print("reward_sum: " + str(reward_sum))

                break

        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            logging.info('Inference call failed. This is normal at the end of training.')