# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import math

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_TREE_GYM_ENV = os.getenv('MINERL_TREE_GYM_ENV', 'MineRLTreechop-v0')
MINERL_NAVI_GYM_ENV = os.getenv('MINERL_NAVI_GYM_ENV', 'MineRLNavigate-v0')
MINERL_STONE_GYM_ENV = os.getenv('MINERL_STONE_GYM_ENV', 'MineRLObtainIronPickaxe-v0')
MINERL_IRON_GYM_ENV = os.getenv('MINERL_IRON_GYM_ENV', 'MineRLObtainIronPickaxe-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
#MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/media/kimbring2/6224AA7924AA5039/minerl_data')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser_tree = Parser('performance/',
                allowed_environment=MINERL_TREE_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

parser_navi = Parser('performance/',
                allowed_environment=MINERL_NAVI_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

parser_stone = Parser('performance/',
                allowed_environment=MINERL_STONE_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

parser_iron = Parser('performance/',
                allowed_environment=MINERL_IRON_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)


class cnn_rnn_network():
    def __init__(self, H=None, scope=None, act_num=None):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(shape=[None,64,64,3], dtype=tf.float32, name='state')
            self.conv1 = slim.conv2d(inputs=self.state, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', 
                                biases_initializer=None)
            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4,4], stride=[2,2], padding='VALID', 
                                biases_initializer=None)
            self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding='VALID', 
                                biases_initializer=None)
            self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=2048, kernel_size=[4,4],stride=[1,1], padding='VALID',
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
    # How to sample minerl data is document here:
    # http://minerl.io/docs/tutorials/data_sampling.html
    H = 2048
    train_env = "iron"

    root_path = '/home/kimbring2/Desktop/competition_submission_starter_template/'
    model_path = root_path + 'train/'
    summary_path = root_path + 'train_summary/'

    if (train_env == "treechop"):
        tree_network = cnn_rnn_network(scope='treechop', act_num=15, H=H)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tree_model_path = model_path + 'MineRLTreechop-v0'
        tree_summary_path = summary_path + 'MineRLTreechop-v0'

        variables = tf.trainable_variables(scope=None)
        variables_tree_restore = [v for v in variables if v.name.split('/')[0] in ['treechop']]
        
        saver_tree = tf.train.Saver(variables_tree_restore, max_to_keep=5)

        #print('Loading Model...')
        #saver = tf.train.Saver(variables_restore)
        #ckpt = tf.train.get_checkpoint_state(model_path)
        #saver.restore(sess, ckpt.model_checkpoint_path)

        merged = tf.summary.merge_all()
        tree_train_writer = tf.summary.FileWriter(tree_summary_path, sess.graph)
        MINERL_TREE_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/media/kimbring2/6224AA7924AA5039/minerl_data/')
        data_tree = minerl.data.make(MINERL_TREE_GYM_ENV, data_dir=MINERL_TREE_DATA_ROOT)

        # Sample code for illustration, add your training code below
        #env = gym.make(MINERL_GYM_ENV)

        episode_count = 0
        for current_state, action, reward, next_state, done in data_tree.sarsd_iter(num_epochs=5000, max_sequence_len=800):
            length = (current_state['pov'].shape)[0]
            #print("length: " + str(length))
            if (length != 800):
                continue

            train_length = length

            action_list = []
            states_list = []
            for i in range(0, length):
                state_concat = current_state['pov'][i].astype(np.float32) / 255.0 - 0.5
      
                camera_threshols = (abs(action['camera'][i][0]) + abs(action['camera'][i][1])) / 2.0
                #print("camera_threshols: " + str(camera_threshols))
                if (camera_threshols > 2.5):
                    if ( (action['camera'][i][1] < 0) & ( abs(action['camera'][i][0]) < abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action['camera'][i][1] > 0) & ( abs(action['camera'][i][0]) < abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action['camera'][i][0] < 0) & ( abs(action['camera'][i][0]) > abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action['camera'][i][0] > 0) & ( abs(action['camera'][i][0]) > abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                elif (action['forward'][i] == 1):
                    if (action['attack'][i] == 0):
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    else:
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                elif ( (action['right'][i] == 1) | (action['left'][i] == 1) ):
                    if ( (action['right'][i] == 0) & (action['left'][i] == 1) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    elif ( (action['right'][i] == 1) & (action['left'][i] == 0) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                else:                   
                    if (action['attack'][i] == 0):
                        train_length = train_length - 1
                        continue
                    else:
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  
                states_list.append(state_concat)
                action_list.append(action_)
            
            batch_size = divmod(train_length, 10)[0]            
            #print("batch_size: " + str(batch_size))  
            state_train = (np.zeros([batch_size,H]),np.zeros([batch_size,H]))
            feed_dict = {tree_network.state:np.stack(states_list[0:batch_size*10], 0),
                         tree_network.real_action:np.stack(action_list[0:batch_size*10], 0),
                         tree_network.trainLength:10,
                         tree_network.state_in:state_train,
                         tree_network.batch_size:batch_size
                        }
            
            #if episode_count % 100 == 0:
            summary, _ = sess.run([merged, tree_network.train_step], feed_dict=feed_dict)
            tree_train_writer.add_summary(summary, episode_count)

            sess.run(tree_network.train_step, feed_dict=feed_dict)
            print("episode_count: " + str(episode_count))
            if episode_count % 10 == 0:
                saver_tree.save(sess, tree_model_path + '/tree_model-' + str(episode_count) + '.cptk')
                print("Saved Tree Model")

            print("episode_count: " + str(episode_count))
            episode_count = episode_count + 1
    elif (train_env == "navigate"):
        navi_network = cnn_rnn_network(scope='navigate', act_num=15, H=H)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        navi_model_path = model_path + 'MineRLNavigate-v0'
        navi_summary_path = summary_path + 'MineRLNavigate-v0'

        variables = tf.trainable_variables(scope=None)
        variables_navi_restore = [v for v in variables if v.name.split('/')[0] in ['navigate']]
        saver_navi = tf.train.Saver(variables_navi_restore, max_to_keep=5)

        merged = tf.summary.merge_all()
        navi_train_writer = tf.summary.FileWriter(navi_summary_path, sess.graph)
        MINERL_NAVI_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/media/kimbring2/6224AA7924AA5039/minerl_data/')
        data_navi = minerl.data.make(MINERL_NAVI_GYM_ENV, data_dir=MINERL_NAVI_DATA_ROOT)

        episode_count = 0
        for current_state, action, reward, next_state, done in data_navi.sarsd_iter(num_epochs=1000, max_sequence_len=800):
            length = (current_state['pov'].shape)[0]
            if (length != 800):
                continue

            train_length = length

            action_list = []
            states_list = []
            for i in range(0, length):
                state_concat = current_state['pov'][i].astype(np.float32) / 255.0 - 0.5
          
                camera_threshols = (abs(action['camera'][i][0]) + abs(action['camera'][i][1])) / 2.0
                #print("camera_threshols: " + str(camera_threshols))
                if (camera_threshols > 2.5):
                    if ( (action['camera'][i][1] < 0) & ( abs(action['camera'][i][0]) < abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action['camera'][i][1] > 0) & ( abs(action['camera'][i][0]) < abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action['camera'][i][0] < 0) & ( abs(action['camera'][i][0]) > abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action['camera'][i][0] > 0) & ( abs(action['camera'][i][0]) > abs(action['camera'][i][1]) ) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                elif (action['forward'][i] == 1):
                    if (action['attack'][i] == 0):
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    else:
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                elif ( (action['right'][i] == 1) | (action['left'][i] == 1) ):
                    if ( (action['right'][i] == 0) & (action['left'][i] == 1) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    elif ( (action['right'][i] == 1) & (action['left'][i] == 0) ):
                        if (action['attack'][i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                else:
                    if (action['attack'][i] == 0):
                        train_length = train_length - 1
                        continue
                    else:
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

                                                               
                states_list.append(state_concat)
                action_list.append(action_)
            
            batch_size = divmod(train_length, 10)[0]    
            state_train = (np.zeros([batch_size,H]),np.zeros([batch_size,H]))
            feed_dict = {navi_network.state:np.stack(states_list[0:batch_size*10], 0),
                         navi_network.real_action:np.stack(action_list[0:batch_size*10], 0),
                         navi_network.trainLength:10,
                         navi_network.state_in:state_train,
                         navi_network.batch_size:batch_size
                        }
            
            #if episode_count % 100 == 0:
            summary, _ = sess.run([merged, navi_network.train_step], feed_dict=feed_dict)
            navi_train_writer.add_summary(summary, episode_count)

            sess.run(navi_network.train_step, feed_dict=feed_dict)
            #print("episode_count: " + str(episode_count))
            if episode_count % 10 == 0:
                saver_navi.save(sess, navi_model_path + '/navi_model-' + str(episode_count) + '.cptk')
                print("Saved Navi Model")

            print("episode_count: " + str(episode_count))
            episode_count = episode_count + 1
    elif (train_env == "stone"):
        stone_network = cnn_rnn_inventory_network(scope='stone', act_num=26, H=H)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        variables = tf.trainable_variables(scope=None)
        variables_stone_restore = [v for v in variables if v.name.split('/')[0] in ['stone']]
        saver_stone = tf.train.Saver(variables_stone_restore, max_to_keep=5)
        stone_model_path = model_path + 'MineRLObtainIronPickaxe-v0'
        stone_summary_path = summary_path + 'MineRLObtainIronPickaxe-v0'

        merged = tf.summary.merge_all()
        stone_train_writer = tf.summary.FileWriter(stone_summary_path, sess.graph)
        MINERL_STONE_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/media/kimbring2/6224AA7924AA5039/minerl_data/')
        data_stone = minerl.data.make(MINERL_STONE_GYM_ENV, data_dir=MINERL_STONE_DATA_ROOT)

        episode_count = 0
        for current_state, action, reward, next_state, done in data_stone.sarsd_iter(num_epochs=5000, max_sequence_len=3000):
            length = (current_state['pov'].shape)[0]
            #print("length: " + str(length))
            if (length != 3000):
                continue

            result = np.where(current_state['equipped_items']['mainhand']['type'] == 3)
            #print('result[0]: ', result[0])
            #print("current_state['equipped_items']['mainhand']['type'][result]: " + str(current_state['equipped_items']['mainhand']['type'][result]))

            train_length = len(result[0])
            #print("train_length: " + str(train_length))
            if (train_length < 10):
                continue

            current_pov = current_state['pov'][result]
            current_inventory = current_state['inventory']

            current_coal = current_inventory['coal'][result]
            current_cobblestone = current_inventory['cobblestone'][result]
            current_crafting_table = current_inventory['crafting_table'][result]
            current_dirt = current_inventory['dirt'][result]
            current_furnace = current_inventory['furnace'][result]
            current_iron_axe = current_inventory['iron_axe'][result]
            current_iron_ingot = current_inventory['iron_ingot'][result]
            current_iron_ore = current_inventory['iron_ore'][result]
            current_iron_pickaxe = current_inventory['iron_pickaxe'][result]
            current_log = current_inventory['log'][result]
            current_planks = current_inventory['planks'][result]
            current_stick = current_inventory['stick'][result]
            current_stone = current_inventory['stone'][result]
            current_stone_axe = current_inventory['stone_axe'][result]
            current_stone_pickaxe = current_inventory['stone_pickaxe'][result]
            current_torch = current_inventory['torch'][result]
            current_wooden_axe = current_inventory['wooden_axe'][result]
            current_wooden_pickaxe = current_inventory['wooden_pickaxe'][result]

            action_camera = action['camera'][result]
            action_jump = action['jump'][result]
            action_forward = action['forward'][result]
            action_left = action['left'][result]
            action_right = action['right'][result]
            action_attack = action['attack'][result]
            action_place = action['place'][result]
            action_equip = action['equip'][result]
            action_nearbyCraft = action['nearbyCraft'][result]
            action_craft = action['craft'][result]

            action_list = []
            states_list = []
            for i in range(0, train_length):
                pov = current_pov[i].astype(np.float32) / 255.0 - 0.5

                coal = current_coal[i]
                cobblestone = current_cobblestone[i]
                crafting_table = current_crafting_table[i]
                dirt = current_dirt[i]
                furnace = current_furnace[i]
                iron_axe = current_iron_axe[i]
                iron_ingot = current_iron_ingot[i]
                iron_ore = current_iron_ore[i]
                iron_pickaxe = current_iron_pickaxe[i]
                log = current_log[i]
                planks = current_planks[i]
                stick = current_stick[i]
                stone = current_stone[i]
                stone_axe = current_stone_axe[i]
                stone_pickaxe = current_stone_pickaxe[i]
                torch = current_torch[i]
                wooden_axe = current_wooden_axe[i]
                wooden_pickaxe = current_wooden_pickaxe[i]
          
                coal_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * coal
                cobblestone_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * cobblestone
                crafting_table_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * crafting_table
                dirt_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * dirt
                furnace_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * furnace
                iron_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_axe
                iron_ingot_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_ingot
                iron_ore_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_ore
                iron_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * iron_pickaxe
                log_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * log
                planks_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * planks
                stick_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stick
                stone_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stone
                stone_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stone_axe
                stone_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * stone_pickaxe
                torch_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * torch
                wooden_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * wooden_axe
                wooden_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32) * wooden_pickaxe
                
                state_concat = np.concatenate([pov, coal_channel, cobblestone_channel, crafting_table_channel, dirt_channel, furnace_channel, 
                                               iron_axe_channel, iron_ingot_channel, iron_ore_channel, iron_pickaxe_channel, log_channel, 
                                               planks_channel, stick_channel, stone_channel, stone_axe_channel, stone_pickaxe_channel,
                                               torch_channel, wooden_axe_channel, wooden_pickaxe_channel], axis=-1)
          
                camera_threshols = (abs(action_camera[i][0]) + abs(action_camera[i][1])) / 2.0
                if (camera_threshols > 2.5):
                    if ( (action_camera[i][1] < 0) & ( abs(action_camera[i][0]) < abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ =     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ =     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_camera[i][1] > 0) & ( abs(action_camera[i][0]) < abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ =     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ =     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_camera[i][0] < 0) & ( abs(action_camera[i][0]) > abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ =     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ =     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_camera[i][0] > 0) & ( abs(action_camera[i][0]) > abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ =     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif (action_forward[i] == 1):
                    if (action_attack[i] == 0):
                        action_ =         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    else:
                        action_ =         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif ( (action_right[i] == 1) | (action_left[i] == 1) ):
                    if ( (action_right[i] == 0) & (action_left[i] == 1) ):
                        if (action_attack[i] == 0):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_right[i] == 1) & (action_left[i] == 0) ):
                        if (action_attack[i] == 0):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                else:
                    if (action_attack[i] == 0):
                        if (action_equip[i] == 3):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif (action_equip[i] == 5):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        elif (action_place[i] == 4):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif (action_place[i] == 5):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif (action_place[i] == 6):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

                        elif (action_craft[i] == 1):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                        elif (action_craft[i] == 2):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                        elif (action_craft[i] == 3):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        elif (action_craft[i] == 4):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

                        elif (action_nearbyCraft[i] == 2):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                        elif (action_nearbyCraft[i] == 4):
                            action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                        else:
                            train_length = train_length - 1
                            continue
                    else:
                        action_ =         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                
                #print("len(state_concat): " + str(len(state_concat)))
                #print("len(action_): " + str(len(action_)))

                states_list.append(state_concat)
                action_list.append(action_)
            
            #print("len(states_list): " + str(len(states_list)))
            #print("len(action_list): " + str(len(action_list)))

            #print("len(states_list): " + str(len(states_list)))
            if (len(states_list) < 10):
                continue

            batch_size = divmod(train_length, 10)[0]
            state_train = (np.zeros([batch_size,H]),np.zeros([batch_size,H]))
            feed_dict = {stone_network.state:np.stack(states_list[0:batch_size*10], 0),
                         stone_network.real_action:np.stack(action_list[0:batch_size*10], 0),
                         stone_network.trainLength:10,
                         stone_network.state_in:state_train,
                         stone_network.batch_size:batch_size
                        }
                
            #if episode_count % 100 == 0:
            summary, _ = sess.run([merged, stone_network.train_step], feed_dict=feed_dict)
            stone_train_writer.add_summary(summary, episode_count)

            sess.run(stone_network.train_step, feed_dict=feed_dict)
            #print("episode_count: " + str(episode_count))
            if episode_count % 10 == 0:
                saver_stone.save(sess, stone_model_path + '/stone_model-' + str(episode_count) + '.cptk')
                print("Saved Stone Model")

            print("episode_count: " + str(episode_count))
            episode_count = episode_count + 1    
    elif (train_env == "iron"):
        H_iron = 512
        iron_network = cnn_rnn_inventory_network(scope='iron', act_num=26, H=H_iron)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        variables = tf.trainable_variables(scope=None)
        variables_iron_restore = [v for v in variables if v.name.split('/')[0] in ['iron']]
        saver_iron = tf.train.Saver(variables_iron_restore, max_to_keep=5)
        iron_model_path = model_path + 'MineRLObtainIronPickaxe-v0'
        iron_summary_path = summary_path + 'MineRLObtainIronPickaxe-v0'

        merged = tf.summary.merge_all()
        stone_train_writer = tf.summary.FileWriter(iron_summary_path, sess.graph)
        MINERL_IRON_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/media/kimbring2/6224AA7924AA5039/minerl_data/')
        data_iron = minerl.data.make(MINERL_IRON_GYM_ENV, data_dir=MINERL_IRON_DATA_ROOT)

        episode_count = 0
        for current_state, action, reward, next_state, done in data_iron.sarsd_iter(num_epochs=5000, max_sequence_len=2000):
            length = (current_state['pov'].shape)[0]
            #print("length: " + str(length))
            if (length != 2000):
                continue

            #result = np.where(current_state['equipped_items']['mainhand']['type'] == 3)
            #print('result[0]: ', result[0])
            #print("current_state['equipped_items']['mainhand']['type'][result]: " + str(current_state['equipped_items']['mainhand']['type'][result]))

            #train_length = len(result[0])
            #print("train_length: " + str(train_length))
            #if (train_length < 10):
            #    continue

            current_pov = current_state['pov']
            current_inventory = current_state['inventory']

            action_camera = action['camera']
            action_jump = action['jump']
            action_forward = action['forward']
            action_left = action['left']
            action_right = action['right']
            action_attack = action['attack']
            action_place = action['place']
            action_equip = action['equip']
            action_nearbyCraft = action['nearbyCraft']
            action_craft = action['craft']
            action_list = []
            states_list = []
            for i in range(0, length):
                pov = current_pov[i].astype(np.float32) / 255.0 - 0.5
                inventory = current_inventory

                coal = inventory['coal'][i]
                cobblestone = inventory['cobblestone'][i]
                crafting_table = inventory['crafting_table'][i]
                dirt = inventory['dirt'][i]
                furnace = inventory['furnace'][i]
                iron_axe = inventory['iron_axe'][i]
                iron_ingot = inventory['iron_ingot'][i]
                iron_ore = inventory['iron_ore'][i]
                iron_pickaxe = inventory['iron_pickaxe'][i]
                log = inventory['log'][i]
                planks = inventory['planks'][i]
                stick = inventory['stick'][i]
                stone = inventory['stone'][i]
                stone_axe = inventory['stone_axe'][i]
                stone_pickaxe = inventory['stone_pickaxe'][i]
                torch = inventory['torch'][i]
                wooden_axe = inventory['wooden_axe'][i]
                wooden_pickaxe = inventory['wooden_pickaxe'][i]
          
                coal_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*coal/2304.0
                cobblestone_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*cobblestone/2304.0
                crafting_table_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*crafting_table/2304.0
                dirt_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*dirt/2304.0
                furnace_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*furnace/2304.0
                iron_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*iron_axe/2304.0
                iron_ingot_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*iron_ingot/2304.0
                iron_ore_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*iron_ore/2304.0
                iron_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*iron_pickaxe/2304.0
                log_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*log/2304.0
                planks_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*planks/2304.0
                stick_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*stick/2304.0
                stone_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*stone/2304.0
                stone_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*stone_axe/2304.0
                stone_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*stone_pickaxe/2304.0
                torch_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*torch/2304.0
                wooden_axe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*wooden_axe/2304.0
                wooden_pickaxe_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*wooden_pickaxe/2304.0
                
                state_concat = np.concatenate([pov, coal_channel, cobblestone_channel, crafting_table_channel, dirt_channel, furnace_channel, 
                                               iron_axe_channel, iron_ingot_channel, iron_ore_channel, iron_pickaxe_channel, log_channel, 
                                               planks_channel, stick_channel, stone_channel, stone_axe_channel, stone_pickaxe_channel,
                                               torch_channel, wooden_axe_channel, wooden_pickaxe_channel], axis=-1)

                camera_threshols = (abs(action_camera[i][0]) + abs(action_camera[i][1])) / 2.0
                if (camera_threshols > 2.5):
                    if ( (action_camera[i][1] < 0) & ( abs(action_camera[i][0]) < abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_camera[i][1] > 0) & ( abs(action_camera[i][0]) < abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_camera[i][0] < 0) & ( abs(action_camera[i][0]) > abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_camera[i][0] > 0) & ( abs(action_camera[i][0]) > abs(action_camera[i][1]) ) ):
                        if (action_attack[i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif (action_forward[i] == 1):
                    if (action_attack[i] == 0):
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    else:
                        action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif ( (action_right[i] == 1) | (action_left[i] == 1) ):
                    if ( (action_right[i] == 0) & (action_left[i] == 1) ):
                        if (action_attack[i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif ( (action_right[i] == 1) & (action_left[i] == 0) ):
                        if (action_attack[i] == 0):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        else:
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                else:
                    if (action_attack[i] == 0):
                        if (action_place[i] == 0):
                            length = length - 1
                            continue

                        elif (action_equip[i] == 3):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif (action_equip[i] == 5):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        elif (action_place[i] == 4):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif (action_place[i] == 5):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                        elif (action_place[i] == 6):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

                        elif (action_craft[i] == 1):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                        elif (action_craft[i] == 2):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                        elif (action_craft[i] == 3):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                        elif (action_craft[i] == 4):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

                        elif (action_nearbyCraft[i] == 2):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                        elif (action_nearbyCraft[i] == 4):
                            action_ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                    else:
                        action_ =     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                                                               
                states_list.append(state_concat)
                action_list.append(action_)
            
            #print("len(states_list): " + str(len(states_list)))
            #if (len(states_list) < 10):
            #    continue

            batch_size = divmod(length, 10)[0]
            state_train = (np.zeros([batch_size,H_iron]),np.zeros([batch_size,H_iron]))
            feed_dict = {iron_network.state:np.stack(states_list[0:batch_size*10], 0),
                         iron_network.real_action:np.stack(action_list[0:batch_size*10], 0),
                         iron_network.trainLength:10,
                         iron_network.state_in:state_train,
                         iron_network.batch_size:batch_size
                        }
                
            #if episode_count % 100 == 0:
            summary, _ = sess.run([merged, iron_network.train_step], feed_dict=feed_dict)
            stone_train_writer.add_summary(summary, episode_count)

            sess.run(iron_network.train_step, feed_dict=feed_dict)
            #print("episode_count: " + str(episode_count))
            if episode_count % 10 == 0:
                saver_iron.save(sess, iron_model_path + '/iron_model-' + str(episode_count) + '.cptk')
                print("Saved Iron Model")

            print("episode_count: " + str(episode_count))
            episode_count = episode_count + 1

    # Save trained model to train/ directory
    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    env.close()

if __name__ == "__main__":
    main()