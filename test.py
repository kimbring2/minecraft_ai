# Simple env test.
import json
import select
import time
import logging
import os
import random

import aicrowd_helper
import gym
import minerl

import coloredlogs
coloredlogs.install(logging.DEBUG)

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import math

from env_wrappers import ContinuingTimeLimitMonitor

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
#MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechopDebug-v0')
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainIronPickaxe-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 100))


class cnn_rnn_inventory_network():
    def __init__(self, H=None, scope=None, act_num=None):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None,64,64,21], dtype=tf.float32, name='state')
            self.conv1 = slim.conv2d(inputs=self.state, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', 
                                biases_initializer=None)
            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', 
                                biases_initializer=None)
            self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', 
                                biases_initializer=None)
            self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=H, kernel_size=[4,4],stride=[1,1], padding='VALID',
                                biases_initializer=None)
            #print("slim.flatten(self.conv4): " + str(slim.flatten(self.conv4)))

            self.trainLength = tf.placeholder(dtype=tf.int32, name='trainLength')
            #We take the output from the final convolutional layer and send it to a recurrent layer.
            #The input must be reshaped into [batch x trace x units] for rnn processing, 
            #and then returned to [batch x units] when sent through the upper levles.
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
            self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size,self.trainLength,H])
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=H, state_is_tuple=True, name='rnn_cell')
            self.state_in = self.rnn_cell.zero_state(self.batch_size, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=self.rnn_cell, dtype=tf.float32,
                                                   initial_state=self.state_in)
            self.rnn = tf.reshape(self.rnn, shape=[-1,H], name='rnn')
            W = tf.get_variable(shape=[H,act_num], initializer=tf.contrib.layers.xavier_initializer(), name='W')
            self.score = tf.matmul(self.rnn, W, name='score')
            self.probability = tf.nn.softmax(self.score, name='probability')
            self.real_action = tf.placeholder(shape=[None,act_num], dtype=tf.int32, name='real_action')
            
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.real_action, logits=self.score))
            tf.summary.scalar('loss', self.loss)
            self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


def main():
    """
    This function will be called for training phase.
    """
    # Sample code for illustration, add your code below to run in test phase.
    # Load trained model from train/ directory
    env = gym.make(MINERL_GYM_ENV)
    root_path = '/home/kimbring2/Desktop/competition_submission_starter_template/'
    env = ContinuingTimeLimitMonitor(env, root_path + 'monitor', mode='evaluation', 
                                     video_callable=lambda episode_id: True, force=True)

    root_path = '/home/kimbring2/Desktop/competition_submission_starter_template/'

    H_Iron = 512
    iron_network = cnn_rnn_inventory_network(scope='iron', act_num=26, H=H_Iron)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    iron_path = './train/MineRLObtainIronPickaxe-v0/'

    variables = tf.trainable_variables(scope=None)
    variables_iron_restore = [v for v in variables if v.name.split('/')[0] in ['iron']]

    print('Loading Iron Model...')
    saver_iron = tf.train.Saver(variables_iron_restore)
    ckpt_iron = tf.train.get_checkpoint_state(iron_path)
    saver_iron.restore(sess, ckpt_iron.model_checkpoint_path)

    for i in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = env.reset()
        done = False
        net_reward = 0

        pre_net_reward = net_reward
        while not done:
            pov = obs['pov'].astype(np.float32) / 255.0 - 0.5
            inventory = obs['inventory']
            
            coal = inventory['coal']
            cobblestone = inventory['cobblestone']
            crafting_table = inventory['crafting_table']
            dirt = inventory['dirt']
            furnace = inventory['furnace']
            iron_axe = inventory['iron_axe']
            iron_ingot = inventory['iron_ingot']
            iron_ore = inventory['iron_ore']
            iron_pickaxe = inventory['iron_pickaxe']
            log = inventory['log']
            planks = inventory['planks']
            stick = inventory['stick']
            stone = inventory['stone']
            stone_axe = inventory['stone_axe']
            stone_pickaxe = inventory['stone_pickaxe']
            torch = inventory['torch']
            wooden_axe = inventory['wooden_axe']
            wooden_pickaxe = inventory['wooden_pickaxe']
            
            coal_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * coal / 2304.0
            cobblestone_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * cobblestone / 2304.0
            crafting_table_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * crafting_table / 2304.0
            dirt_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * dirt / 2304.0
            furnace_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * furnace / 2304.0
            iron_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_axe / 2304.0
            iron_ingot_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_ingot / 2304.0
            iron_ore_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_ore / 2304.0
            iron_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_pickaxe / 2304.0
            log_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * log / 2304.0
            planks_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * planks / 2304.0
            stick_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stick / 2304.0
            stone_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stone / 2304.0
            stone_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stone_axe / 2304.0
            stone_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stone_pickaxe / 2304.0
            torch_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * torch / 2304.0
            wooden_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * wooden_axe / 2304.0
            wooden_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * wooden_pickaxe / 2304.0
                
            state_concat = np.concatenate([pov, coal_channel, cobblestone_channel, crafting_table_channel, dirt_channel, furnace_channel, 
                                           iron_axe_channel, iron_ingot_channel, iron_ore_channel, iron_pickaxe_channel, log_channel, 
                                           planks_channel, stick_channel, stone_channel, stone_axe_channel, stone_pickaxe_channel,
                                           torch_channel, wooden_axe_channel, wooden_pickaxe_channel], axis=-1)

            equip_type = obs['equipped_items']['mainhand']['type']
            #print("equip_type: " + str(equip_type))

            state_iron = (np.zeros([1, H_Iron]),np.zeros([1, H_Iron]))
            iron_action_probability = sess.run(iron_network.probability, feed_dict={iron_network.state:[state_concat],
                                                                                    iron_network.trainLength:1,
                                                                                    iron_network.state_in:state_iron,
                                                                                    iron_network.batch_size:1}
                                               )
            
            e = 0.02
            if np.random.rand(1) >= e:
                action_index = np.argmax(iron_action_probability)
            else:
                action_index = random.randint(0,26)
                
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
                action['attack'] = 1

            elif (action_index == 8):
                action['forward'] = 1
                action['attack'] = 0
            elif (action_index == 9):
                action['forward'] = 1
                action['attack'] = 1

            elif (action_index == 10):
                action['left'] = 1
                action['right'] = 0
                action['attack'] = 0
            elif (action_index == 11):
                action['left'] = 1
                action['right'] = 0
                action['attack'] = 1
            elif (action_index == 12):
                action['left'] = 0
                action['right'] = 1
                action['attack'] = 0
            elif (action_index == 13):
                action['left'] = 0
                action['right'] = 1
                action['attack'] = 1

            elif (action_index == 14):
                action['equip'] = 3
            elif (action_index == 15):
                action['equip'] = 5

            elif (action_index == 16):
                action['place'] = 4
            elif (action_index == 17):
                action['place'] = 5
            elif (action_index == 18):
                action['place'] = 6

            elif (action_index == 19):
                action['craft'] = 1
            elif (action_index == 20):
                action['craft'] = 2
            elif (action_index == 21):
                action['craft'] = 3
            elif (action_index == 22):
                action['craft'] = 4

            elif (action_index == 23):
                action['nearbyCraft'] = 2
            elif (action_index == 24):
                action['nearbyCraft'] = 4

            elif (action_index == 25):
                action['attack'] = 1
                    
            #action['left'] = 0
            #action['right'] = 0
            #action['sprint'] = 0

            obs1, reward, done, info = env.step(action)

            obs = obs1
            net_reward += reward

            if done == True:
                break

        print("Total reward: ", net_reward)
    env.close()

if __name__ == "__main__":
    main()
