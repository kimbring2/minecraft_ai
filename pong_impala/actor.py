import grpc
import tensorflow as tf
import numpy as np
import collections
import cv2
import gym

import argparse
from absl import flags
from absl import logging

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='Pong implementation')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
arguments = parser.parse_args()


env = gym.make('Pong-v0')

client = grpc.Client("localhost:8686")
EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')

run_id = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=3, dtype=np.int64)


for i in range(0, 2000000):
	obs = env.reset()
	obs = obs[35:195:2, ::2,:]
	obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

	obs_t = np.stack((obs, obs, obs, obs), axis=2)

	done = False
	reward = 0.0
	reward_sum = 0
	episode_step = 0
	while True:
		try:
			if arguments.env_id == 0:
				env.render()

			obs_t_reshaped = np.reshape(obs_t, (1,80,80,4))

			# EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')
			env_output = EnvOutput(np.array([reward], dtype=np.float32), np.array([done]), 
														 obs_t_reshaped, np.array([False]), np.array([episode_step], dtype=np.int32))

			# client.inference(env_id, run_id, env_output, raw_reward)
			action = client.inference(np.array([arguments.env_id], dtype=np.int32), np.array([run_id[0]], dtype=np.int64), 
																env_output, np.array([reward], dtype=np.float32))

			obs1, reward, done, info = env.step(action)
			#reward = reward
			obs1 = obs1[35:195:2, ::2,:]
			obs1 = cv2.cvtColor(obs1, cv2.COLOR_BGR2GRAY)
			obs1 = np.reshape(obs1, (80, 80, 1))

			obs_t1 = np.append(obs1, obs_t[:, :, :3], axis=2)

			obs_t = obs_t1
			obs = obs1
			reward_sum += reward
			if done:
				print("reward_sum: " + str(reward_sum))
				break

			episode_step += 1
		except (tf.errors.UnavailableError, tf.errors.CancelledError):
			logging.info('Inference call failed. This is normal at the end of training.')
