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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='MineRL IMPALA Actor')
parser.add_argument('--env_id', type=int, default=0, help='ID of environment')
arguments = parser.parse_args()

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:" + str(5555 + arguments.env_id))


def render(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imshow('obs_' + str(arguments.env_id), obs)
    cv2.waitKey(1)


scores = []
episodes = []
average = []

def PlotModel(score, episode):
    scores.append(score)
    episodes.append(episode)
    average.append(sum(scores[-50:]) / len(scores[-50:]))

    plt.plot(episodes, average)
    plt.xlabel('episodes')
    plt.ylabel('average scores')

    try:
        plt.savefig("reward_graph.png")
    except OSError:
        pass

    return average[-1]


env = gym.make('MineRLNavigateDense-v0')

EnvOutput = collections.namedtuple('EnvOutput', 'reward done observation abandoned episode_step')

run_id = np.random.randint(low=0, high=np.iinfo(np.int64).max, size=3, dtype=np.int64)

for episode_step in range(0, 2000000):
	obs = env.reset()
	#print("obs['compassAngle'].shape: ", obs.keys())

	pov_array = obs['pov'] / 255.0
	compassAngle_array = obs['compassAngle'] / 360.0
	compassAngle_array = np.ones((64,64,1)) * compassAngle_array

	state_array = np.concatenate((pov_array, compassAngle_array), 2)
	state_array = np.expand_dims(state_array, 0)

	#state_array_t = np.stack((state_array, state_array, state_array, state_array), axis=2)

	done = False
	reward = 0.0
	reward_sum = 0
	while True:
		try:
			#print("step")
			if arguments.env_id == 0:
				render(obs['pov'])

			state_array_reshaped = np.reshape(state_array, (1,64,64,4)) 

			env_output = {"env_id": np.array([arguments.env_id]), 
                          "reward": reward,
                          "done": done, 
                          "observation": state_array_reshaped}
			socket.send_pyobj(env_output)
			action_index = int(socket.recv_pyobj()['action'])

			action = env.action_space.noop()
			if (action_index == 0):
				action['camera'] = [0, -2.5]
			elif (action_index == 1):
				action['camera'] = [0, 2.5]
			elif (action_index == 2):
				action['forward'] = 1
				action['jump'] = 1

			obs1, reward, done, info = env.step(action)
			#print("reward: ", reward)

			next_pov_array = obs1['pov'] / 255.0
			next_compassAngle_array = obs1['compassAngle'] / 360.0
			next_compassAngle_array = np.ones((64,64,1)) * next_compassAngle_array

			next_state_array = np.concatenate((next_pov_array, next_compassAngle_array), 2)
			next_state_array = np.expand_dims(next_state_array, 0)
			#next_state_array = np.reshape(next_state_array, (64, 64, 1))

			#state_array_t1 = np.append(next_state_array, state_array_t[:, :, :3], axis=2)

			state_array = next_state_array
			obs = obs1
			reward_sum += reward
			if done:
				if arguments.env_id == 0:
					average_reward = PlotModel(reward_sum, episode_step)
					print("average_reward: " + str(average_reward))
				else:
					print("reward_sum: " + str(reward_sum))

				#print("reward_sum: " + str(reward_sum))
				break

		except (tf.errors.UnavailableError, tf.errors.CancelledError):
			logging.info('Inference call failed. This is normal at the end of training.')