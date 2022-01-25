import os
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTM, Reshape, LSTMCell
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2
import threading
from threading import Thread, Lock
import time
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tfd = tfp.distributions


class OurModel(tf.keras.Model):
    def __init__(self, input_shape, action_space):
        super(OurModel, self).__init__()
        
        self.flatten = Flatten()
        #self.conv_1 = Conv2D(32, 8, 3, padding="valid", activation="relu")
        #self.conv_2 = Conv2D(64, 4, 2, padding="valid", activation="relu")
        #self.conv_3 = Conv2D(64, 3, 1, padding="valid", activation="relu")
        #self.conv_4 = Conv2D(512, 7, 1, padding="valid", activation="relu")
        self.dense_0 = Dense(512, activation='relu')
        self.lstm = LSTMCell(256)
        self.dense_1 = Dense(action_space)
        self.dense_2 = Dense(1)
        self.dense_3 = Dense(256, activation='relu')
        
    def call(self, X_input, memory_state, carry_state):
        batch_size = X_input.shape[0]
        
        #conv_1 = self.conv_1(X_input)
        #conv_2 = self.conv_2(conv_1)
        #conv_3 = self.conv_3(conv_2)
        #conv_4 = self.conv_4(conv_3)
        X_input = self.flatten(X_input)
        X_input = self.dense_0(X_input)
        #conv_4_flattened = Flatten()(conv_4)
        
        initial_state = (memory_state, carry_state)
        LSTM_output, lstm_state = self.lstm(X_input, initial_state)
        
        final_memory_state = lstm_state[0]
        final_carry_state =  lstm_state[1]

        LSTM_output_flattened = Flatten()(LSTM_output)
        #LSTM_output_flattened = self.dense_3(LSTM_output_flattened)
        
        action_logit = self.dense_1(LSTM_output_flattened)
        value = self.dense_2(LSTM_output_flattened)
        
        return action_logit, value, final_memory_state, final_carry_state


def safe_log(x):
  """Computes a safe logarithm which returns 0 if x is zero."""
  return tf.where(
      tf.math.equal(x, 0),
      tf.zeros_like(x),
      tf.math.log(tf.math.maximum(1e-12, x)))


def take_vector_elements(vectors, indices):
    """
    For a batch of vectors, take a single vector component
    out of each vector.
    Args:
      vectors: a [batch x dims] Tensor.
      indices: an int32 Tensor with `batch` entries.
    Returns:
      A Tensor with `batch` entries, one for each vector.
    """
    return tf.gather_nd(vectors, tf.stack([tf.range(tf.shape(vectors)[0]), indices], axis=1))


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
mse_loss = tf.keras.losses.MeanSquaredError()


class A3CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = 2000000, 0, -21.0 # specific for pong
        self.lock = Lock()
        self.lr = 0.000025

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4

        # Instantiate plot memory
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.env_name, self.lr)
        self.model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.ActorCritic = OurModel(input_shape=self.state_size, action_space=self.action_size)
        
        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
    
    def act(self, state, memory_state, carry_state):
        memory_state = tf.constant(memory_state, tf.float32)
        carry_state = tf.constant(carry_state, tf.float32)
        
        # Use the network to predict the next action to take, using the model
        prediction = self.ActorCritic(state, memory_state, carry_state, training=False)
        action = tf.random.categorical(prediction[0], 1).numpy()

        memory_state = prediction[2].numpy()
        carry_state = prediction[3].numpy()
        
        return action[0][0], memory_state, carry_state

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0

            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation

        return discounted_r
    
    def replay(self, states, actions, rewards, memory_states, carry_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        
        memory_states = np.vstack(memory_states)
        carry_states = np.vstack(carry_states)
        
        batch_size = states.shape[0]
        
        # Compute discounted rewards
        discounted_r = self.discount_rewards(rewards)
        discounted_r_ = np.vstack(discounted_r)
        with tf.GradientTape() as tape:
            action_logits = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            
            memory_state = tf.expand_dims(memory_states[0], 0)
            carry_state = tf.expand_dims(carry_states[0], 0)
            for i in tf.range(0, batch_size):
                prediction = self.ActorCritic(tf.expand_dims(states[i], 0), 
                                              memory_state, carry_state, training=True)
                
                action_logits = action_logits.write(i, prediction[0][0])
                values = values.write(i, prediction[1][0])
                
                memory_state = prediction[2]
                carry_state = prediction[3]
                
            action_logits = action_logits.stack()
            values = values.stack()
            
            action_logits_selected = take_vector_elements(action_logits, actions)
            
            advantages = discounted_r - np.stack(values)[:, 0] 
            
            action_logits_selected = tf.nn.softmax(action_logits_selected)
            action_logits_selected_probs = tf.math.log(action_logits_selected)
            
            action_logits_ = tf.nn.softmax(action_logits)
            #action_logits_ = tf.math.log(action_logits_)
            dist = tfd.Categorical(probs=action_logits_)
            action_log_prob = dist.prob(actions)
            action_log_prob = tf.math.log(action_log_prob)
            #print("action_logits_selected_probs: ", action_logits_selected_probs)
            #print("action_log_prob.shape: ", action_log_prob)
            
            actor_loss = -tf.math.reduce_mean(action_log_prob * advantages) 
            #actor_loss = tf.cast(actor_loss, 'float32')
            
            action_probs = tf.nn.softmax(action_logits)
            #entropy_loss = tf.keras.losses.categorical_crossentropy(action_logits_probs, action_logits_probs)
            #actor_loss = sparse_ce(actions, action_probs, sample_weight=advantages)
            
            critic_loss_ = huber_loss(values, discounted_r)
            critic_loss = mse_loss(values, discounted_r_)
            critic_loss = tf.cast(critic_loss, 'float32')
            #print("critic_loss: ", critic_loss)
            total_loss = actor_loss + critic_loss
        
        #print("total_loss: ", total_loss)
        #print("")
            
        grads = tape.gradient(total_loss, self.ActorCritic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.ActorCritic.trainable_variables))
        
    def load(self, model_name):
        self.ActorCritic = load_model(model_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.ActorCritic.save(self.model_name)
        #self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path + ".png")
            except OSError:
                pass

        return self.average[-1]
    
    def reset(self, env):
        state = env.reset()
        
        state = state[35:195:2, ::2,:]
        state = 0.299*state[:,:,0] + 0.587*state[:,:,1] + 0.114*state[:,:,2]
        state[state < 100] = 0
        state[state >= 100] = 255
        state = np.array(state).astype(np.float32) / 255.0

        return state
    
    def step(self, action, env):
        next_state, reward, done, info = env.step(action)
        
        return next_state, reward, done, info
    
    def train(self, n_threads):
        self.env.close()
        # Instantiate one environment per thread
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # Create threads
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self, envs[i], i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()
            
        for t in threads:
            time.sleep(10)
            t.join()
            
    def train_threading(self, agent, env, thread):
        while self.episode < self.EPISODES:
            # Reset episode
            score, done, SAVING = 0, False, ''
            state = self.reset(env)
            #obs_t = np.stack((state, state, state, state), axis=1)
            #print("obs_t.shape: ", obs_t.shape)

            states, actions, rewards = [], [], []
            memory_states, carry_states = [], []
            
            memory_state = np.zeros([1,256], dtype=np.float32)
            carry_state = np.zeros([1,256], dtype=np.float32)
            while not done:
                action, memory_state, carry_state = agent.act(np.reshape(state, (1,80,80,1)), memory_state, carry_state)

                next_state, reward, done, _ = self.step(action, env)

                next_state = next_state[35:195:2, ::2,:]
                next_state = 0.299*next_state[:,:,0] + 0.587*next_state[:,:,1] + 0.114*next_state[:,:,2]
                next_state[next_state < 100] = 0
                next_state[next_state >= 100] = 255
                next_state = np.array(next_state).astype(np.float32) / 255.0
                next_state = np.reshape(next_state, (80,80,1))
 
                states.append(np.reshape(state, (1,80,80,1)))
                actions.append(action)
                rewards.append(reward)
                memory_states.append(memory_state)
                carry_states.append(carry_state)

                #obs_t = obs_t1

                score += reward
                state = next_state
                    
            self.lock.acquire()
            self.replay(states, actions, rewards, memory_states, carry_states)
            self.lock.release()
            
            states, actions, rewards = [], [], []
                    
            # Update episode count
            with self.lock:
                average = self.PlotModel(score, self.episode)
                # saving best models
                if average >= self.max_average:
                    self.max_average = average
                    #self.save()
                    SAVING = "SAVING"
                else:
                    SAVING = ""

                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, thread, score, average, SAVING))
                if(self.episode < self.EPISODES):
                    self.episode += 1

        env.close()            

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break

        self.env.close()


if __name__ == "__main__":
    env_name = 'PongDeterministic-v4'
    #env_name = 'Pong-v0'
    agent = A3CAgent(env_name)
    
    #agent.run() # use as A2C
    agent.train(n_threads=1) # use as A3C
    #agent.test('Models/Pong-v0_A3C_2.5e-05_Actor.h5', '')