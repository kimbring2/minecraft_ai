import tensorflow as tf
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""
  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.num_actions = num_actions
    
    self.conv_1 = layers.Conv2D(16, 8, 4, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_2 = layers.Conv2D(32, 4, 2, padding="valid", activation="relu", kernel_regularizer='l2')
    self.conv_3 = layers.Conv2D(32, 3, 1, padding="valid", activation="relu", kernel_regularizer='l2')
    
    self.lstm = layers.LSTM(128, return_sequences=True, return_state=True, kernel_regularizer='l2')
    
    self.common = layers.Dense(num_hidden_units, activation="relu", kernel_regularizer='l2')
    self.actor = layers.Dense(num_actions, kernel_regularizer='l2')
    self.critic = layers.Dense(1, kernel_regularizer='l2')

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_actions': self.num_actions,
        'num_hidden_units': self.num_hidden_units
    })
    return config
    
  def call(self, inputs: tf.Tensor, memory_state: tf.Tensor, carry_state: tf.Tensor, training) -> Tuple[tf.Tensor, tf.Tensor, 
                                                                                                        tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(inputs)[0]

    conv_1 = self.conv_1(inputs)
    conv_2 = self.conv_2(conv_1)
    conv_3 = self.conv_3(conv_2)
    #print("conv_3.shape: ", conv_3.shape)
    conv_3_reshaped = layers.Reshape((4*4,32))(conv_3)
    
    initial_state = (memory_state, carry_state)
    lstm_output, final_memory_state, final_carry_state  = self.lstm(conv_3_reshaped, initial_state=initial_state, 
                                                                    training=training)
    
    X_input = layers.Flatten()(lstm_output)
    #print("X_input.shape: ", X_input.shape)
    x = self.common(X_input)
    
    return self.actor(x), self.critic(x), final_memory_state, final_carry_state