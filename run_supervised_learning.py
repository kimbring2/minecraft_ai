import collections
import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import glob
import random
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple
import gym
import minerl
import os
import gym
import minerl
import numpy as np
from absl import flags
import argparse
import network

parser = argparse.ArgumentParser(description='Minecraft Supervised Learning')
parser.add_argument('--workspace_path', type=str, help='root directory of project')
parser.add_argument('--data_path', type=str, help='root directory of dataset')
parser.add_argument('--gpu_use', type=bool, default=False, help='use gpu')

arguments = parser.parse_args()

if arguments.gpu_use == True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


workspace_path = arguments.workspace_path

writer = tf.summary.create_file_writer(workspace_path + "/tensorboard")

tree_data = minerl.data.make('MineRLTreechop-v0', data_dir=arguments.data_path)

class TrajetoryDataset(tf.data.Dataset):
    def _generator(num_trajectorys):
        while True:
            trajectory_names = tree_data.get_trajectory_names()
            #print("len(trajectory_names): ", len(trajectory_names))
            
            trajectory_name = random.choice(trajectory_names)
            print("trajectory_name: ", trajectory_name)
            
            trajectory = tree_data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
            #print("trajectory: ", trajectory)
            
            all_actions = []
            all_obs = []
            for dataset_observation, dataset_action, reward, next_state, done in trajectory:  
                #state_pov = dataset_observation['pov']
                
                inventory_channel = np.zeros((64,64,1))
                if 'inventory' in dataset_observation:
                    region_max_height = dataset_observation['pov'].shape[0]
                    region_max_width = dataset_observation['pov'].shape[1]
                    rs = 8
                    if min(region_max_height, region_max_width) < rs:
                        raise ValueError("'region_size' is too large.")
                    num_element_width = region_max_width // rs

                    inventory_channel = np.zeros(shape=list(dataset_observation['pov'].shape[:-1]) + [1], 
                                                 dtype=dataset_observation['pov'].dtype)
                    #print("state['inventory'].keys(): " + str(state['inventory'].keys()))
                    for key_idx, key in enumerate(dataset_observation['inventory'].keys()):
                        #print("key.shape : " + str(key))
                        #print("state['inventory'][key][i] : " + str(state['inventory'][key][i]))
                        item_scaled = np.clip(1 - 1 / (dataset_observation['inventory'][key] + 1),  # Inversed
                                                0, 1)
                        #print("item_scaled : " + str(item_scaled))
                        item_channel = np.ones(shape=[rs, rs, 1], dtype=dataset_observation['pov'].dtype) * item_scaled
                        width_low = (key_idx % num_element_width) * rs
                        height_low = (key_idx // num_element_width) * rs

                        if height_low + rs > region_max_height:
                            raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component.")

                        inventory_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel

                observation = np.concatenate((dataset_observation['pov'] / 255.0, inventory_channel), axis=2)

                action_camera_0 = dataset_action['camera'][0]
                action_camera_1 = dataset_action['camera'][1]
                action_attack = dataset_action['attack']
                action_forward = dataset_action['forward']
                action_jump = dataset_action['jump']
                action_back = dataset_action['back']
                action_left = dataset_action['left']
                action_right = dataset_action['right']
                action_sneak = dataset_action['sneak']

                camera_threshols = (abs(action_camera_0) + abs(action_camera_1)) / 2.0
                if (camera_threshols > 2.5):
                    if ( (action_camera_1 < 0) & ( abs(action_camera_0) < abs(action_camera_1) ) ):
                        if (action_attack == 0):
                            action_index = 0
                        else:
                            action_index = 1
                    elif ( (action_camera_1 > 0) & ( abs(action_camera_0) < abs(action_camera_1) ) ):
                        if (action_attack == 0):
                            action_index = 2
                        else:
                            action_index = 3
                    elif ( (action_camera_0 < 0) & ( abs(action_camera_0) > abs(action_camera_1) ) ):
                        if (action_attack == 0):
                            action_index = 4
                        else:
                            action_index = 5
                    elif ( (action_camera_0 > 0) & ( abs(action_camera_0) > abs(action_camera_1) ) ):
                        if (action_attack == 0):
                            action_index = 6
                        else:
                            action_index = 7

                elif (action_forward == 1):
                    if (action_attack == 0):
                        action_index = 8
                    else:
                        action_index = 9
                elif (action_jump == 1):
                    if (action_attack == 0):
                        action_index = 10
                    else:
                        action_index = 11
                elif (action_back == 1):
                    if (action_attack == 0):
                        action_index = 12
                    else:
                        action_index = 13
                elif (action_left == 1):
                    if (action_attack == 0):
                        action_index = 14
                    else:
                        action_index = 15
                elif (action_right == 1):
                    if (action_attack == 0):
                        action_index = 16
                    else:
                        action_index = 17
                elif (action_sneak == 1):
                    if (action_attack == 0):
                        action_index = 18
                    else:
                        action_index = 19
                else:
                    continue

                if (dataset_action['attack'] == 0 and dataset_action['back'] == 0 and dataset_action['camera'][0] == 0.0 and 
                    dataset_action['camera'][1] == 0.0 and dataset_action['forward'] == 0 and dataset_action['jump'] == 0 and 
                    dataset_action['left'] == 0 and dataset_action['right'] == 0 and dataset_action['sneak'] == 0):
                    #print("continue: ")
                    continue

                #print("observation.shape: ", observation.shape)
                #print("action_index: ", action_index)
                #print("done: ", done)

                all_obs.append(observation)
                all_actions.append(np.array([action_index]))

            print("len(all_obs): ", len(all_obs))
            print("")
            yield (all_obs, all_actions)

            break
    
    def __new__(cls, num_trajectorys=3):
      return tf.data.Dataset.from_generator(
          cls._generator,
          output_types=(tf.dtypes.float32, tf.dtypes.int32),
          args=(num_trajectorys,)
    )

dataset = tf.data.Dataset.range(1).interleave(TrajetoryDataset, 
  num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

num_actions = 20
num_hidden_units = 512

#model = tf.keras.models.load_model('MineRL_SL_Model')
model = network.ActorCritic(num_actions, num_hidden_units)
#model.load_weights("model/MineRL_SL_Model")

cce_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.0001)


@tf.function
def supervised_replay(replay_obs_list, replay_act_list, memory_state, carry_state):
    replay_obs_array = tf.concat(replay_obs_list, 0)
    replay_act_array = tf.concat(replay_act_list, 0)
    replay_memory_state_array = tf.concat(memory_state, 0)
    replay_carry_state_array = tf.concat(carry_state, 0)

    memory_state = replay_memory_state_array
    carry_state = replay_carry_state_array

    batch_size = replay_obs_array.shape[0]
    tf.print("batch_size: ", batch_size)
    
    with tf.GradientTape() as tape:
        act_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in tf.range(0, batch_size):
            prediction = model(tf.expand_dims(replay_obs_array[i,:,:,:], 0), memory_state, carry_state, training=True)
            act_pi = prediction[0]
            memory_state = prediction[2]
            carry_state = prediction[3]
        
            act_probs = act_probs.write(i, act_pi[0])

        act_probs = act_probs.stack()

        tf.print("replay_act_array: ", replay_act_array)
        tf.print("tf.argmax(act_probs, 1): ", tf.argmax(act_probs, 1))

        replay_act_array_onehot = tf.one_hot(replay_act_array, num_actions)
        replay_act_array_onehot = tf.reshape(replay_act_array_onehot, (batch_size, num_actions))
        act_loss = cce_loss(replay_act_array_onehot, act_probs)

        #tf.print("act_loss: ", act_loss)
        regularization_loss = tf.reduce_sum(model.losses)
        total_loss = act_loss + 1e-5 * regularization_loss
    
        #tf.print("total_loss: ", total_loss)
        #tf.print("")
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss, memory_state, carry_state


def supervised_train(dataset, training_episode):
    for batch in dataset:
        episode_size = batch[0].shape[1]
        print("episode_size: ", episode_size)
    
        replay_obs_list = batch[0][0]
        replay_act_list = batch[1][0]
     
        memory_state = np.zeros([1,128], dtype=np.float32)
        carry_state =  np.zeros([1,128], dtype=np.float32)
        step_length = 32
        for episode_index in range(0, episode_size, step_length):
            obs = replay_obs_list[episode_index:episode_index+step_length,:,:,:]
            act = replay_act_list[episode_index:episode_index+step_length,:]
            
            #print("len(obs): ", len(obs))
            if len(obs) != step_length:
                break
            
            total_loss, next_memory_state, next_carry_state = supervised_replay(obs, act, memory_state, carry_state)
            memory_state = next_memory_state
            carry_state = next_carry_state
        
            print("total_loss: ", total_loss)
            print("")
            
        with writer.as_default():
            tf.summary.scalar("total_loss", total_loss, step=training_episode)
            writer.flush()

        if training_episode % 100 == 0:
            model.save_weights(workspace_path + '/model/supervised_model_' + str(training_episode))
            
        
for training_episode in range(0, 2000000):
    supervised_train(dataset, training_episode)