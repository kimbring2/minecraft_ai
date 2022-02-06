#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import cv2
import time
import gym
import numpy as np
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

import os
import argparse
from absl import flags
from absl import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Pong IMPALA Client')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
arguments = parser.parse_args()

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + arguments.env_id))
#EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')

#env = gym.make('Pong-v0')
env = gym.make('PongDeterministic-v4')

scores = []
episodes = []
average = []


def PlotModel(score, episode):
    scores.append(score)
    episodes.append(episode)
    average.append(sum(scores[-50:]) / len(scores[-50:]))
    if str(episode)[-2:] == "00":# much faster than episode % 100
        plt.plot(episodes, scores)
        plt.xlabel('episodes')
        plt.ylabel('scores')
        #plt.show()

        #pylab.plot(episodes, scores, 'b')
        #pylab.plot(episodes, average, 'r')
        #pylab.ylabel('Score', fontsize=18)
        #pylab.xlabel('Steps', fontsize=18)
        try:
            #pylab.savefig("reward_graph.png")
            plt.savefig("reward_graph.png")
        except OSError:
            pass

    return average[-1]


average_reward = 0
for episode_step in range(0, 2000000):
    obs = env.reset()
    obs = obs[35:195:2, ::2,:]
    obs = 0.299*obs[:,:,0] + 0.587*obs[:,:,1] + 0.114*obs[:,:,2]
    obs[obs < 100] = 0
    obs[obs >= 100] = 255
    obs = np.array(obs).astype(np.float32) / 255.0

    obs_t = np.stack((obs, obs, obs, obs), axis=2)

    done = False
    reward = 0.0
    reward_sum = 0
    start = time.time()
    while True:
        try:
            #if arguments.env_id == 0:
            #    env.render()

            obs_t_reshaped = np.reshape(obs_t, (80,80,4))

            # EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')
            #env_output = EnvOutput(np.array([reward], dtype=np.float32), np.array([done]), 
            #                                             obs_t_reshaped, np.array([False]), np.array([episode_step], dtype=np.int32))

            # client.inference(env_id, run_id, env_output, raw_reward)
            #action = client.inference(np.array([arguments.env_id], dtype=np.int32), np.array([run_id[0]], dtype=np.int64), 
            #                                                    env_output, np.array([reward], dtype=np.float32))

            env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward / 20.0, 
                          "done": done, 
                          "observation": obs_t_reshaped}
            socket.send_pyobj(env_output)
            action = socket.recv_pyobj()
            #print("action[\"env_id\"]: ", action["env_id"])
            #print("action: ", action)
            #print("env.action_space: ", env.action_space)

            #action = env.action_space.sample()
            obs1, reward, done, info = env.step(int(action['action']))
            obs1 = obs1[35:195:2, ::2,:]
            obs1 = 0.299*obs1[:,:,0] + 0.587*obs1[:,:,1] + 0.114*obs1[:,:,2]
            obs1[obs1 < 100] = 0
            obs1[obs1 >= 100] = 255
            obs1 = np.array(obs1).astype(np.float32) / 255.0
            obs1 = np.reshape(obs1, (80,80,1))
            obs_t1 = np.append(obs1, obs_t[:, :, :3], axis=2)

            obs_t = obs_t1
            obs = obs1
            reward_sum += reward
            if done:
                if arguments.env_id == 0:
                    average_reward = PlotModel(reward_sum, episode_step)
                    print("average_reward: " + str(average_reward))
                else:
                    print("reward_sum: " + str(reward_sum))

                end = time.time()
                elapsed_time = end - start
                #print("elapsed_time: " + str(elapsed_time))
                #episode_step += 1

                break

        except (tf.errors.UnavailableError, tf.errors.CancelledError):
            logging.info('Inference call failed. This is normal at the end of training.')