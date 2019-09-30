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

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
#MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLTreechopDebug-v0')
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 100))

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


def main():
    """
    This function will be called for training phase.
    """
    # Sample code for illustration, add your code below to run in test phase.
    # Load trained model from train/ directory
    env = gym.make(MINERL_GYM_ENV)

    navi_network = cnn_rnn_network(scope='navigate', H=2048, act_num=15)
    tree_network = cnn_rnn_network(scope='treechop', H=2048, act_num=15)
    stone_network = cnn_rnn_network(scope='stone', H=2048, act_num=15)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    navi_path = './train/MineRLNavigate-v0/'
    tree_path = './train/MineRLTreechop-v0/'
    stone_path = './train/MineRLObtainIronPickaxe-v0/'

    variables = tf.trainable_variables(scope=None)
    variables_tree_restore = [v for v in variables if v.name.split('/')[0] in ['treechop']]
    variables_navi_restore = [v for v in variables if v.name.split('/')[0] in ['navigate']]
    variables_stone_restore = [v for v in variables if v.name.split('/')[0] in ['stone']]

    print('Loading Tree Model...')
    saver_tree = tf.train.Saver(variables_tree_restore)
    ckpt_tree = tf.train.get_checkpoint_state(tree_path)
    saver_tree.restore(sess, ckpt_tree.model_checkpoint_path)
    
    print('Loading Stone Model...')
    saver_stone = tf.train.Saver(variables_stone_restore)
    ckpt_stone = tf.train.get_checkpoint_state(stone_path)
    saver_stone.restore(sess, ckpt_stone.model_checkpoint_path)
    
    print('Loading Navi Model...')
    saver_navi = tf.train.Saver(variables_navi_restore)
    ckpt_navi = tf.train.get_checkpoint_state(navi_path)
    saver_navi.restore(sess, ckpt_navi.model_checkpoint_path)

    for i in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = env.reset()
        done = False
        net_reward = 0
        pre_reward = 0

        place_flag = 0
        equip_flag = 0
        random_flag = 0
        time_flag = 0
        navi_flag = 0
        tree_flag = 1
        equip_flag = 0
        wooden_flag = 0
        attack_flag = 0
        back_flag = 0
        log_flag = 0

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
            
            equip_type = obs['equipped_items']['mainhand']['type']
            #print("equip_type: " + str(equip_type))

            H_Navi = 2048
            H_Tree = 2048
            H_Stone = 2048
            state_navi = (np.zeros([1, H_Navi]),np.zeros([1, H_Navi]))
            state_tree = (np.zeros([1, H_Tree]),np.zeros([1, H_Tree]))
            state_stone = (np.zeros([1, H_Stone]),np.zeros([1, H_Stone]))
            
            navi_action_probability = sess.run(navi_network.probability, feed_dict={navi_network.state:[pov],
                                                                                    navi_network.trainLength:1,
                                                                                    navi_network.state_in:state_navi,
                                                                                    navi_network.batch_size:1}
                                               )
            
            tree_action_probability = sess.run(tree_network.probability, feed_dict={tree_network.state:[pov],
                                                                                    tree_network.trainLength:1,
                                                                                    tree_network.state_in:state_tree,
                                                                                    tree_network.batch_size:1}
                                               )
            
            stone_action_probability = sess.run(stone_network.probability, feed_dict={stone_network.state:[pov],
                                                                                      stone_network.trainLength:1,
                                                                                      stone_network.state_in:state_stone,
                                                                                      stone_network.batch_size:1}
                                               )
            
            e = 0.02
            print("tree_flag: " + str(tree_flag))
            print("navi_flag: " + str(navi_flag))
            print("time_flag: " + str(time_flag))
            print("inventory: " + str(inventory))
            #print("equip_type: " + str(equip_type))
            print("")
            if ( ( (log == 0) & (log_flag == 0) & (planks < 4) ) | (equip_type == 'wooden_pickaxe') ):
                if np.random.rand(1) >= e:
                    if ( (tree_flag == 1) & (navi_flag == 0) & (equip_type != 'wooden_pickaxe') ):
                        action_index = np.argmax(tree_action_probability)
                    elif ( (tree_flag == 0) & (navi_flag != 0) & (equip_type != 'wooden_pickaxe') ):
                        action_index = np.argmax(navi_action_probability)
                    elif ( (tree_flag == 0) & (navi_flag == 0) & (equip_type == 'wooden_pickaxe') ):
                        action_index = np.argmax(stone_action_probability)
                        print("obs['equipped_items']['mainhand']: " + str(obs['equipped_items']['mainhand']))

                    #action_index = np.argmax(tree_action_probability)
                else:
                    action_index = random.randint(0,15)
                
                #print("action_index: " + str(action_index))
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
                    action['attack'] = 1

                if (action['forward'] == 1):
                    action['jump'] = 1
            else:
                if (place_flag == 0):
                    action = env.action_space.noop()
                    if ( (planks < 4) & (log != 0) ):
                        action['place'] = 0; action['craft'] = 3; 
                        action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
                        action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
                        action['forward'] = 0; action['jump'] = 0
                    elif (stick < 1):
                        action['place'] = 0; action['craft'] = 2; 
                        action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
                        action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
                        action['forward'] = 0; action['jump'] = 0
                    elif (crafting_table == 0):
                        action['place'] = 0; action['craft'] = 4; 
                        action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
                        action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
                        action['forward'] = 0; action['jump'] = 0


                if ( (crafting_table >= 1) & (stick >= 2) & (planks >= 3) ):
                    action = env.action_space.noop()
                    if (place_flag == 0):
                        action['place'] = 0; action['craft'] = 0; 
                        action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
                        action['attack'] = 0; action['camera'][0] = 30; action['camera'][1] = 0;
                        action['forward'] = 0; action['jump'] = 0;
                        action['equip'] = 0
                        place_flag = place_flag + 1
                    elif (place_flag == 1):
                        action['place'] = 4; action['craft'] = 0; 
                        action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
                        action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
                        action['forward'] = 0; action['jump'] = 0;
                        action['equip'] = 0
                        place_flag = place_flag + 1
                    elif (place_flag == 2):
                        action['place'] = 0; action['craft'] = 0; 
                        action['nearbyCraft'] = 2; action['nearbySmelt'] = 0
                        action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
                        action['forward'] = 0; action['jump'] = 0;
                        action['equip'] = 0
                        place_flag = -1
                
                print("place_flag: " + str(place_flag))
                print("inventory: " + str(inventory))
                print("obs['equipped_items']['mainhand']: " + str(obs['equipped_items']['mainhand']))
                print("action: " + str(action))
            if (wooden_pickaxe >= 1):
                action['place'] = 0; action['craft'] = 0; 
                action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
                action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
                action['forward'] = 0; action['jump'] = 0;
                action['equip'] = 'wooden_pickaxe'
                wooden_flag = wooden_flag + 1
                    
            #action['left'] = 0
            #action['right'] = 0
            action['sprint'] = 0

            if (navi_flag != 0):
                action['attack'] = 0
            
            #if (wooden_flag == 1):
            #    print("obs['equipped_items']['mainhand']: " + str(obs['equipped_items']['mainhand']))
                
            #print("action: " + str(action))
                  
            #if (time_flag % 1000 == 0):
            #    action['camera'][1] = -90

            if (pre_reward == 0):
                time_flag = time_flag + 1

                if ( (time_flag > 2000) & (navi_flag == 0) ):
                    time_flag = 0
                    navi_flag = 1
                    tree_flag = 0
                    action['camera'][1] = -180;
            else:
                time_flag = 0
            
            if (navi_flag != 0):
                navi_flag = navi_flag + 1

            if (navi_flag == 500):
                navi_flag = 0
                time_flag = 0
                tree_flag = 1

            obs1, reward, done, info = env.step(action)

            obs = obs1
            net_reward += reward

            pre_reward = reward

            if done == True:
                break

        print("Total reward: ", net_reward)
    env.close()

if __name__ == "__main__":
    main()
