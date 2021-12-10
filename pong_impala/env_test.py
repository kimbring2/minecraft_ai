import gym
import cv2
import numpy as np 

env = gym.make('Pong-v0')
print(env.observation_space.dtype)
#env.action_space:  Discrete(6)
#env.observation_space:  Box(210, 160, 3)

def imshow(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.imshow("image", image)
	if cv2.waitKey(25) & 0xFF == ord("q"):
		cv2.destroyAllWindows()
		return


for i_episode in range(20):
    obs = env.reset()
    obs = obs[35:195:2, ::2,:]
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    #print("obs.shape: ", obs.shape)

    obs_t = np.stack((obs, obs, obs, obs), axis=2)
    #print("obs_t.shape: ", obs_t.shape)
    for t in range(100):
        print("obs_t.shape: ", obs_t.shape)

        imshow(obs)

        action = env.action_space.sample()
        obs1, reward, done, info = env.step(action)
        obs1 = obs1[35:195:2, ::2,:]
        obs1 = cv2.cvtColor(obs1, cv2.COLOR_BGR2GRAY)
        obs1 = np.reshape(obs1, (80, 80, 1))
        #print("obs1.shape: ", obs1.shape)

        obs_t1 = np.append(obs1, obs_t[:, :, :3], axis=2)

        obs_t = obs_t1
        obs = obs1

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
