# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions/classes."""

import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  



FLAGS = flags.FLAGS


# `observation` is the observation *after* a transition. When `done` is True,
# `observation` will be the observation *after* the reset.
EnvOutput = collections.namedtuple(
    'EnvOutput', 'reward done observation abandoned episode_step')


Settings = collections.namedtuple(
    'Settings', 'strategy inference_devices training_strategy encode decode')


MultiHostSettings = collections.namedtuple(
    'MultiHostSettings', 'strategy hosts training_strategy encode decode')


def init_learner_multi_host(num_training_tpus: int):
  """Performs common learner initialization including multi-host setting.

  In multi-host setting, this function will enter a loop for secondary learners
  until the primary learner signals end of training.

  Args:
    num_training_tpus: Number of training TPUs.

  Returns:
    A MultiHostSettings object.
  """
  tpu = ''
  job_name = None


  if tf.config.experimental.list_logical_devices('TPU'):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu, job_name=job_name)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    assert num_training_tpus % topology.num_tasks == 0
    num_training_tpus_per_task = num_training_tpus // topology.num_tasks

    hosts = []
    training_coordinates = []
    for per_host_coordinates in topology.device_coordinates:
      host = topology.cpu_device_name_at_coordinates(
          per_host_coordinates[0], job=job_name)
      task_training_coordinates = (
          per_host_coordinates[:num_training_tpus_per_task])
      training_coordinates.extend([[c] for c in task_training_coordinates])

      inference_coordinates = per_host_coordinates[num_training_tpus_per_task:]
      hosts.append((host, [
          topology.tpu_device_name_at_coordinates(c, job=job_name)
          for c in inference_coordinates
      ]))

    training_da = tf.tpu.experimental.DeviceAssignment(topology,
                                                       training_coordinates)
    training_strategy = tf.distribute.experimental.TPUStrategy(
        resolver, device_assignment=training_da)
    return MultiHostSettings(strategy, hosts, training_strategy, tpu_encode,
                             tpu_decode)
  else:
    tf.device('/cpu').__enter__()
    any_gpu = tf.config.experimental.list_logical_devices('GPU')
    device_name = '/device:GPU:0' if any_gpu else '/device:CPU:0'
    strategy = tf.distribute.OneDeviceStrategy(device=device_name)
    enc = lambda x: x
    dec = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)
    return MultiHostSettings(
        strategy, [('/cpu', [device_name])], strategy, enc, dec)


def init_learner(num_training_tpus):
  """Performs common learner initialization."""
  settings = init_learner_multi_host(num_training_tpus)
  if len(settings.hosts) != 1:
    raise ValueError(f'Invalid number of hosts: {len(settings.hosts)}')
  return Settings(settings.strategy, settings.hosts[0][1],
                  settings.training_strategy, settings.encode, settings.decode)


class UnrollStore(tf.Module):
  """Utility module for combining individual environment steps into unrolls."""

  def __init__(self,
               num_envs,
               unroll_length,
               timestep_specs,
               num_overlapping_steps=0,
               name='UnrollStore'):
    super(UnrollStore, self).__init__(name=name)
    with self.name_scope:
      self._full_length = num_overlapping_steps + unroll_length + 1

      def create_unroll_variable(spec):
        z = tf.zeros(
            [num_envs, self._full_length] + spec.shape.dims, dtype=spec.dtype)
        return tf.Variable(z, trainable=False, name=spec.name)

      self._unroll_length = unroll_length
      self._num_overlapping_steps = num_overlapping_steps
      self._state = tf.nest.map_structure(create_unroll_variable,
                                          timestep_specs)
      # For each environment, the index into the environment dimension of the
      # tensors in self._state where we should add the next element.
      self._index = tf.Variable(
          tf.fill([num_envs], tf.constant(num_overlapping_steps, tf.int32)),
          trainable=False,
          name='index')

  @property
  def unroll_specs(self):
    return tf.nest.map_structure(lambda v: tf.TensorSpec(v.shape[1:], v.dtype),
                                 self._state)

  @tf.function
  @tf.Module.with_name_scope
  def append(self, env_ids, values):
    """Appends values and returns completed unrolls.

    Args:
      env_ids: 1D tensor with the list of environment IDs for which we append
        data.
        There must not be duplicates.
      values: Values to add for each environment. This is a structure
        (in the tf.nest sense) of tensors following "timestep_specs", with a
        batch front dimension which must be equal to the length of 'env_ids'.

    Returns:
      A pair of:
        - 1D tensor of the environment IDs of the completed unrolls.
        - Completed unrolls. This is a structure of tensors following
          'timestep_specs', with added front dimensions: [num_completed_unrolls,
          num_overlapping_steps + unroll_length + 1].
    """
    tf.debugging.assert_equal(
        tf.shape(env_ids),
        tf.shape(tf.unique(env_ids)[0]),
        message=f'Duplicate environment ids in store {self.name}')
    
    tf.nest.map_structure(
        lambda s: tf.debugging.assert_equal(
            tf.shape(env_ids)[0],
            tf.shape(s)[0],
            message=(f'Batch dimension must equal the number of environments '
                     f'in store {self.name}.')),
        values)
    

    curr_indices = self._index.sparse_read(env_ids)
    unroll_indices = tf.stack([env_ids, curr_indices], axis=-1)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_nd_update(unroll_indices, v)

    # Intentionally not protecting against out-of-bounds to make it possible to
    # detect completed unrolls.
    self._index.scatter_add(tf.IndexedSlices(1, env_ids))

    return self._complete_unrolls(env_ids)

  @tf.function
  @tf.Module.with_name_scope
  def reset(self, env_ids):
    """Resets state.

    Note, this is only intended to be called when environments need to be reset
    after preemptions. Not at episode boundaries.

    Args:
      env_ids: The environments that need to have their state reset.
    """
    self._index.scatter_update(
        tf.IndexedSlices(self._num_overlapping_steps, env_ids))

    # The following code is the equivalent of:
    # s[env_ids, :j] = 0
    j = self._num_overlapping_steps
    repeated_env_ids = tf.reshape(
        tf.tile(tf.expand_dims(tf.cast(env_ids, tf.int64), -1), [1, j]), [-1])

    repeated_range = tf.tile(tf.range(j, dtype=tf.int64),
                             [tf.shape(env_ids)[0]])
    indices = tf.stack([repeated_env_ids, repeated_range], axis=-1)

    for s in tf.nest.flatten(self._state):
      z = tf.zeros(tf.concat([tf.shape(repeated_env_ids),
                              s.shape[2:]], axis=0), s.dtype)
      s.scatter_nd_update(indices, z)

  def _complete_unrolls(self, env_ids):
    # Environment with unrolls that are now complete and should be returned.
    env_indices = self._index.sparse_read(env_ids)
    env_ids = tf.gather(
        env_ids,
        tf.where(tf.equal(env_indices, self._full_length))[:, 0])
    env_ids = tf.cast(env_ids, tf.int64)
    unrolls = tf.nest.map_structure(lambda s: s.sparse_read(env_ids),
                                    self._state)

    # Store last transitions as the first in the next unroll.
    # The following code is the equivalent of:
    # s[env_ids, :j] = s[env_ids, -j:]
    j = self._num_overlapping_steps + 1
    repeated_start_range = tf.tile(tf.range(j, dtype=tf.int64),
                                   [tf.shape(env_ids)[0]])
    repeated_end_range = tf.tile(
        tf.range(self._full_length - j, self._full_length, dtype=tf.int64),
        [tf.shape(env_ids)[0]])
    repeated_env_ids = tf.reshape(
        tf.tile(tf.expand_dims(env_ids, -1), [1, j]), [-1])
    start_indices = tf.stack([repeated_env_ids, repeated_start_range], -1)
    end_indices = tf.stack([repeated_env_ids, repeated_end_range], -1)

    for s in tf.nest.flatten(self._state):
      s.scatter_nd_update(start_indices, s.gather_nd(end_indices))

    self._index.scatter_update(
        tf.IndexedSlices(1 + self._num_overlapping_steps, env_ids))

    return env_ids, unrolls


class PrioritizedReplay(tf.Module):
  """Prioritized Replay Buffer.

  This buffer is not threadsafe. Make sure you call insert() and sample() from a
  single thread.
  """

  def __init__(self, size, specs, importance_sampling_exponent,
               name='PrioritizedReplay'):
    super(PrioritizedReplay, self).__init__(name=name)
    self._priorities = tf.Variable(tf.zeros([size]), dtype=tf.float32)
    self._buffer = tf.nest.map_structure(
        lambda ts: tf.Variable(tf.zeros([size] + ts.shape, dtype=ts.dtype)),
        specs)
    self.num_inserted = tf.Variable(0, dtype=tf.int64)
    self._importance_sampling_exponent = importance_sampling_exponent

  @tf.function
  @tf.Module.with_name_scope
  def insert(self, values, priorities):
    """FIFO insertion/removal.

    Args:
      values: The batched values to insert. The tensors must be of the same
        shape and dtype as the `specs` provided in the constructor, except
        including a batch dimension.
      priorities: <float32>[batch_size] tensor with the priorities of the
        elements we insert.
    Returns:
      The indices of the inserted values.
    """
    tf.nest.assert_same_structure(values, self._buffer)
    values = tf.nest.map_structure(tf.convert_to_tensor, values)
    append_size = tf.nest.flatten(values)[0].shape[0]
    start_index = self.num_inserted
    end_index = start_index + append_size

    # Wrap around insertion.
    size = self._priorities.shape[0]
    insert_indices = tf.range(start_index, end_index) % size
    tf.nest.map_structure(
        lambda b, v: b.batch_scatter_update(  
            tf.IndexedSlices(v, insert_indices)),
        self._buffer,
        values)
    self.num_inserted.assign_add(append_size)

    self._priorities.batch_scatter_update(
        tf.IndexedSlices(priorities, insert_indices))

    return insert_indices

  @tf.function
  @tf.Module.with_name_scope
  def sample(self, num_samples, priority_exp):
    r"""Samples items from the replay buffer, using priorities.

    Args:
      num_samples: int, number of replay items to sample.
      priority_exp: Priority exponent. Every item i in the replay buffer will be
        sampled with probability:
         priority[i] ** priority_exp /
             sum(priority[j] ** priority_exp, j \in [0, num_items))
        Set this to 0 in order to get uniform sampling.

    Returns:
      Tuple of:
        - indices: An int64 tensor of shape [num_samples] with the indices in
          the replay buffer of the sampled items.
        - weights: A float32 tensor of shape [num_samples] with the normalized
          weights of the sampled items.
        - sampled_values: A nested structure following the spec passed in the
          contructor, where each tensor has an added front batch dimension equal
          to 'num_samples'.
    """
    tf.debugging.assert_greater_equal(
        self.num_inserted,
        tf.constant(0, tf.int64),
        message='Cannot sample if replay buffer is empty')
    size = self._priorities.shape[0]
    limit = tf.minimum(tf.cast(size, tf.int64), self.num_inserted)
    if priority_exp == 0:
      indices = tf.random.uniform([num_samples], maxval=limit, dtype=tf.int64)
      weights = tf.ones_like(indices, dtype=tf.float32)
    else:
      prob = self._priorities[:limit]**priority_exp
      prob /= tf.reduce_sum(prob)
      indices = tf.random.categorical([tf.math.log(prob)], num_samples)[0]
      # Importance weights.
      weights = (((1. / tf.cast(limit, tf.float32)) /
                  tf.gather(prob, indices)) **
                 self._importance_sampling_exponent)
      weights /= tf.reduce_max(weights)  # Normalize.

    sampled_values = tf.nest.map_structure(
        lambda b: b.sparse_read(indices), self._buffer)
    return indices, weights, sampled_values

  @tf.function
  @tf.Module.with_name_scope
  def update_priorities(self, indices, priorities):
    """Updates the priorities of the items with the given indices.

    Args:
      indices: <int64>[batch_size] tensor with the indices of the items to
        update. If duplicate indices are provided, the priority that will be set
        among possible ones is not specified.
      priorities: <float32>[batch_size] tensor with the new priorities.
    """

    self._priorities.batch_scatter_update(tf.IndexedSlices(priorities, indices))


class HindsightExperienceReplay(PrioritizedReplay):
  """Replay Buffer with Hindsight Experience Replay.

  Hindsight goals are sampled uniformly from subsequent steps in the
  same window (`future` strategy from https://arxiv.org/pdf/1707.01495).
  They are not guaranteed to come from the same episode.

  This buffer is not threadsafe. Make sure you call insert() and sample() from a
  single thread.
  """

  def __init__(self, size, specs, importance_sampling_exponent,
               compute_reward_fn,
               unroll_length,
               substitution_probability,
               name='HindsightExperienceReplay'):
    super(HindsightExperienceReplay, self).__init__(
        size, specs, importance_sampling_exponent, name)
    self._compute_reward_fn = compute_reward_fn
    self._unroll_length = unroll_length
    self._substitution_probability = substitution_probability

  @tf.Module.with_name_scope
  def sample(self, num_samples, priority_exp):
    indices, weights, sampled_values = super(
        HindsightExperienceReplay, self).sample(num_samples, priority_exp)

    observation = sampled_values.env_outputs.observation
    batch_size, time_horizon = observation['achieved_goal'].shape[:2]

    def compute_goal_reward():
      # reward[batch][time] is the reward on transition from timestep time-1
      # to time. This function outputs incorrect rewards for the last transition
      # in each episode but we filter such cases later.
      goal_reward = self._compute_reward_fn(
          achieved_goal=observation['achieved_goal'][:, 1:],
          desired_goal=observation['desired_goal'][:, :-1])
      return tf.concat(values=[goal_reward[:, :1] * np.nan, goal_reward],
                       axis=1)

    # Substitute goals.
    old_goal_reward = compute_goal_reward()
    assert old_goal_reward.shape == observation['achieved_goal'].shape[:-1]
    goal_ind = tf.concat(
        values=[tf.random.uniform((batch_size, 1), min(t + 1, time_horizon - 1),
                                  time_horizon, dtype=tf.int32)
                for t in range(time_horizon)], axis=1)
    substituted_goal = tf.gather(observation['achieved_goal'],
                                 goal_ind, axis=1, batch_dims=1)
    mask = tf.cast(tfp.distributions.Bernoulli(
        probs=self._substitution_probability *
        tf.ones(goal_ind.shape)).sample(), observation['desired_goal'].dtype)
    # We don't substitute goals for the last states in each episodes because we
    # don't store the next states for them.
    mask *= tf.cast(~sampled_values.env_outputs.done,
                    observation['desired_goal'].dtype)
    mask = mask[..., tf.newaxis]
    observation['desired_goal'] = (
        mask * substituted_goal + (1 - mask) * observation['desired_goal'])

    # Substitude reward
    new_goal_reward = compute_goal_reward()
    assert new_goal_reward.shape == observation['achieved_goal'].shape[:-1]
    sampled_values = sampled_values._replace(
        env_outputs=sampled_values.env_outputs._replace(
            reward=sampled_values.env_outputs.reward +
            (new_goal_reward - old_goal_reward) * tf.cast(
                ~sampled_values.env_outputs.done, tf.float32)
            ))

    # Subsample unrolls of length unroll_length + 1.
    assert time_horizon >= self._unroll_length + 1

    unroll_begin_ind = tf.random.uniform(
        (batch_size,), 0, time_horizon - self._unroll_length, dtype=tf.int32)
    unroll_inds = unroll_begin_ind[:, tf.newaxis] + tf.math.cumsum(
        tf.ones((batch_size, self._unroll_length + 1), tf.int32),
        axis=1, exclusive=True)
    subsampled_values = tf.nest.map_structure(
        lambda t: tf.gather(t, unroll_inds, axis=1, batch_dims=1),
        sampled_values)
    if hasattr(sampled_values, 'agent_state'):  # do not subsample the state
      subsampled_values = subsampled_values._replace(
          agent_state=sampled_values.agent_state)

    return indices, weights, subsampled_values


class Aggregator(tf.Module):
  """Utility module for keeping state for individual environments."""

  def __init__(self, num_envs, specs, name='Aggregator'):
    """Inits an Aggregator.

    Args:
      num_envs: int, number of environments.
      specs: Structure (as defined by tf.nest) of tf.TensorSpecs that will be
        stored for each environment.
      name: Name of the scope for the operations.
    """
    super(Aggregator, self).__init__(name=name)
    def create_variable(spec):
      z = tf.zeros([num_envs] + spec.shape.dims, dtype=spec.dtype)
      return tf.Variable(z, trainable=False, name=spec.name)

    self._state = tf.nest.map_structure(create_variable, specs)

  @tf.Module.with_name_scope
  def reset(self, env_ids):
    """Fills the tensors for the given environments with zeros."""
    with tf.name_scope('Aggregator_reset'):
      for s in tf.nest.flatten(self._state):
        s.scatter_update(tf.IndexedSlices(0, env_ids))

  @tf.Module.with_name_scope
  def add(self, env_ids, values):
    """In-place adds values to the state associated to the given environments.

    Args:
      env_ids: 1D tensor with the environment IDs we want to add values to.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'env_ids', or
        should not exist (in which case, the value is broadcasted to all
        environment ids).
    """
    tf.nest.assert_same_structure(values, self._state)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_add(tf.IndexedSlices(v, env_ids))

  @tf.Module.with_name_scope
  def read(self, env_ids):
    """Reads the values corresponding to a list of environments.

    Args:
      env_ids: 1D tensor with the list of environment IDs we want to read.

    Returns:
      A structure of tensors with the same shapes as the input specs. A
      dimension is added in front of each tensor, with size equal to the number
      of env_ids provided.
    """
    return tf.nest.map_structure(lambda s: s.sparse_read(env_ids),
                                 self._state)

  @tf.Module.with_name_scope
  def replace(self, env_ids, values, debug_op_name='', debug_tensors=None):
    """Replaces the state associated to the given environments.

    Args:
      env_ids: 1D tensor with the list of environment IDs.
      values: A structure of tensors following the input spec, with an added
        first dimension that must either have the same size as 'env_ids', or
        should not exist (in which case, the value is broadcasted to all
        environment ids).
      debug_op_name: Debug name for the operation.
      debug_tensors: List of tensors to print when the assert fails.
    """
    tf.debugging.assert_rank(
        env_ids, 1,
        message=f'Invalid rank for aggregator {self.name}')
    tf.debugging.Assert(
        tf.reduce_all(tf.equal(
            tf.shape(env_ids), tf.shape(tf.unique(env_ids)[0]))),
        data=[env_ids,
              (f'Duplicate environment ids in Aggregator: {self.name} with '
               f'op name "{debug_op_name}"')] + (debug_tensors or []),
        summarize=4096,
        name=f'assert_no_dups_{self.name}')
    tf.nest.assert_same_structure(values, self._state)
    for s, v in zip(tf.nest.flatten(self._state), tf.nest.flatten(values)):
      s.scatter_update(tf.IndexedSlices(v, env_ids))


class ProgressLogger(object):
  """Helper class for performing periodic logging of the training progress."""

  def __init__(self,
               summary_writer=None,
               initial_period=0.1,
               period_factor=1.01,
               max_period=10.0,
               starting_step=0):
    """Constructs ProgressLogger.

    Args:
      summary_writer: Tensorflow summary writer to use.
      initial_period: Initial logging period in seconds
        (how often logging happens).
      period_factor: Factor by which logging period is
        multiplied after each iteration (exponential back-off).
      max_period: Maximal logging period in seconds
        (the end of exponential back-off).
      starting_step: Step from which to start the summary writer.
    """
    # summary_writer, last_log_{time, step} are set in reset() function.
    self.summary_writer = None
    self.last_log_time = None
    self.last_log_step = 0
    self.period = initial_period
    self.period_factor = period_factor
    self.max_period = max_period
    # Array of strings with names of values to be logged.
    self.log_keys = []
    self.log_keys_set = set()
    self.step_cnt = tf.Variable(-1, dtype=tf.int64)
    self.ready_values = tf.Variable([-1.0],
                                    dtype=tf.float32,
                                    shape=tf.TensorShape(None))
    self.logger_thread = None
    self.logging_callback = None
    self.terminator = None
    self.reset(summary_writer, starting_step)

  def reset(self, summary_writer=None, starting_step=0):
    """Resets the progress logger.

    Args:
      summary_writer: Tensorflow summary writer to use.
      starting_step: Step from which to start the summary writer.
    """
    self.summary_writer = summary_writer
    self.step_cnt.assign(starting_step)
    self.ready_values.assign([-1.0])
    self.last_log_time = timeit.default_timer()
    self.last_log_step = starting_step

  def start(self, logging_callback=None):
    assert self.logger_thread is None
    self.logging_callback = logging_callback
    self.terminator = threading.Event()
    self.logger_thread = threading.Thread(target=self._logging_loop)
    self.logger_thread.start()

  def shutdown(self):
    assert self.logger_thread
    self.terminator.set()
    self.logger_thread.join()
    self.logger_thread = None

  def log_session(self):
    return []

  def log(self, session, name, value):
    # this is a python op so it happens only when this tf.function is compiled
    if name not in self.log_keys_set:
      self.log_keys.append(name)
      self.log_keys_set.add(name)
    # this is a TF op.
    session.append(value)

  def log_session_from_dict(self, dic):
    session = self.log_session()
    for key in dic:
      self.log(session, key, dic[key])
    return session

  def step_end(self, session, strategy=None, step_increment=1):
    logs = []
    for value in session:
      if strategy:
        value = tf.reduce_mean(tf.cast(
            strategy.experimental_local_results(value)[0], tf.float32))
      logs.append(value)
    self.ready_values.assign(logs)
    self.step_cnt.assign_add(step_increment)

  def _log(self):
    """Perform single round of logging."""
    logging_time = timeit.default_timer()
    step_cnt = self.step_cnt.read_value()
    if step_cnt == self.last_log_step:
      return
    values = self.ready_values.read_value().numpy()
    if values[0] == -1:
      return
    assert len(values) == len(
        self.log_keys
    ), 'Mismatch between number of keys and values to log: %r vs %r' % (
        values, self.log_keys)
    if self.summary_writer:
      self.summary_writer.set_as_default()
    tf.summary.experimental.set_step(step_cnt.numpy())
    if self.logging_callback:
      self.logging_callback()
    for key, value in zip(self.log_keys, values):
      tf.summary.scalar(key, value)
    dt = logging_time - self.last_log_time
    df = tf.cast(step_cnt - self.last_log_step, tf.float32)
    tf.summary.scalar('speed/steps_per_sec', df / dt)
    self.last_log_time, self.last_log_step = logging_time, step_cnt

  def _logging_loop(self):
    """Loop being run in a separate thread."""
    last_log_try = timeit.default_timer()
    while not self.terminator.isSet():
      try:
        self._log()
      except Exception:  
        logging.fatal('Logging failed.', exc_info=True)
      now = timeit.default_timer()
      elapsed = now - last_log_try
      last_log_try = now
      self.period = min(self.period_factor * self.period,
                        self.max_period)
      self.terminator.wait(timeout=max(0, self.period - elapsed))


class StructuredFIFOQueue(tf.queue.FIFOQueue):
  """A tf.queue.FIFOQueue that supports nests and tf.TensorSpec."""

  def __init__(self,
               capacity,
               specs,
               shared_name=None,
               name='structured_fifo_queue'):
    self._specs = specs
    self._flattened_specs = tf.nest.flatten(specs)
    dtypes = [ts.dtype for ts in self._flattened_specs]
    shapes = [ts.shape for ts in self._flattened_specs]
    super(StructuredFIFOQueue, self).__init__(capacity, dtypes, shapes)

  def dequeue(self, name=None):
    result = super(StructuredFIFOQueue, self).dequeue(name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def dequeue_many(self, batch_size, name=None):
    result = super(StructuredFIFOQueue, self).dequeue_many(
        batch_size, name=name)
    return tf.nest.pack_sequence_as(self._specs, result)

  def enqueue(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue(
        tf.nest.flatten(vals), name=name)

  def enqueue_many(self, vals, name=None):
    tf.nest.assert_same_structure(vals, self._specs)
    return super(StructuredFIFOQueue, self).enqueue_many(
        tf.nest.flatten(vals), name=name)


def batch_apply(fn, inputs):
  """Folds time into the batch dimension, runs fn() and unfolds the result.

  Args:
    fn: Function that takes as input the n tensors of the tf.nest structure,
      with shape [time*batch, <remaining shape>], and returns a tf.nest
      structure of batched tensors.
    inputs: tf.nest structure of n [time, batch, <remaining shape>] tensors.

  Returns:
    tf.nest structure of [time, batch, <fn output shape>]. Structure is
    determined by the output of fn.
  """
  time_to_batch_fn = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
  batched = tf.nest.map_structure(time_to_batch_fn, inputs)
  output = fn(*batched)
  prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
  batch_to_time_fn = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())
  return tf.nest.map_structure(batch_to_time_fn, output)


def make_time_major(x):
  """Transposes the batch and time dimensions of a nest of Tensors.

  If an input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A nest of Tensors.

  Returns:
    x transposed along the first two dimensions.
  """

  def transpose(t):  
    t_static_shape = t.shape
    if t_static_shape.rank is not None and t_static_shape.rank < 2:
      return t

    t_rank = tf.rank(t)
    t_t = tf.transpose(t, tf.concat(([1, 0], tf.range(2, t_rank)), axis=0))
    t_t.set_shape(
        tf.TensorShape([t_static_shape[1],
                        t_static_shape[0]]).concatenate(t_static_shape[2:]))
    return t_t

  return tf.nest.map_structure(
      lambda t: tf.xla.experimental.compile(transpose, [t])[0], x)


class TPUEncodedUInt8Spec(tf.TypeSpec):
  """Type specification for composite tensor TPUEncodedUInt8."""

  def __init__(self, encoded_shape, original_shape):
    self._value_specs = (tf.TensorSpec(encoded_shape, tf.uint32),)
    self.original_shape = original_shape

  @property
  def _component_specs(self):
    return self._value_specs

  def _to_components(self, value):
    return (value.encoded,)

  def _from_components(self, components):
    return TPUEncodedUInt8(components[0], self.original_shape)

  def _serialize(self):
    return self._value_specs[0].shape, self.original_shape

  def _to_legacy_output_types(self):
    return self._value_specs[0].dtype

  def _to_legacy_output_shapes(self):
    return self._value_specs[0].shape

  @property
  def value_type(self):
    return TPUEncodedUInt8


class TPUEncodedUInt8(composite_tensor.CompositeTensor):

  def __init__(self, encoded, shape):
    self.encoded = encoded
    self.original_shape = shape
    self._spec = TPUEncodedUInt8Spec(encoded.shape, tf.TensorShape(shape))

  @property
  def _type_spec(self):
    return self._spec


tensor_conversion_registry.register_tensor_conversion_function(
    TPUEncodedUInt8, lambda value, *unused_args, **unused_kwargs: value.encoded)


class TPUEncodedF32Spec(tf.TypeSpec):
  """Type specification for composite tensor TPUEncodedF32Spec."""

  def __init__(self, encoded_shape, original_shape):
    self._value_specs = (tf.TensorSpec(encoded_shape, tf.float32),)
    self.original_shape = original_shape

  @property
  def _component_specs(self):
    return self._value_specs

  def _to_components(self, value):
    return (value.encoded,)

  def _from_components(self, components):
    return TPUEncodedF32(components[0], self.original_shape)

  def _serialize(self):
    return self._value_specs[0].shape, self.original_shape

  def _to_legacy_output_types(self):
    return self._value_specs[0].dtype

  def _to_legacy_output_shapes(self):
    return self._value_specs[0].shape

  @property
  def value_type(self):
    return TPUEncodedF32


class TPUEncodedF32(composite_tensor.CompositeTensor):

  def __init__(self, encoded, shape):
    self.encoded = encoded
    self.original_shape = shape
    self._spec = TPUEncodedF32Spec(encoded.shape, tf.TensorShape(shape))

  @property
  def _type_spec(self):
    return self._spec


tensor_conversion_registry.register_tensor_conversion_function(
    TPUEncodedF32, lambda value, *unused_args, **unused_kwargs: value.encoded)


def num_divisible(v, m):
  return sum([1 for x in v if x % m == 0])


def tpu_encode(ts):
  """Encodes a nest of Tensors in a suitable way for TPUs.

  TPUs do not support tf.uint8, tf.uint16 and other data types. Furthermore,
  the speed of transfer and device reshapes depend on the shape of the data.
  This function tries to optimize the data encoding for a number of use cases.

  Should be used on CPU before sending data to TPU and in conjunction with
  `tpu_decode` after the data is transferred.

  Args:
    ts: A tf.nest of Tensors.

  Returns:
    A tf.nest of encoded Tensors.
  """

  def visit(t):  
    num_elements = t.shape.num_elements()
    # We need a multiple of 128 elements: encoding reduces the number of
    # elements by a factor 4 (packing uint8s into uint32s), and first thing
    # decode does is to reshape with a 32 minor-most dimension.
    if (t.dtype == tf.uint8 and num_elements is not None and
        num_elements % 128 == 0):
      # For details of these transformations, see b/137182262.
      x = tf.xla.experimental.compile(
          lambda x: tf.transpose(x, list(range(1, t.shape.rank)) + [0]), [t])[0]
      x = tf.reshape(x, [-1, 4])
      x = tf.bitcast(x, tf.uint32)
      x = tf.reshape(x, [-1])
      return TPUEncodedUInt8(x, t.shape)
    elif t.dtype == tf.uint8:
      logging.warning('Inefficient uint8 transfer with shape: %s', t.shape)
      return tf.cast(t, tf.bfloat16)
    elif t.dtype == tf.uint16:
      return tf.cast(t, tf.int32)
    elif (t.dtype == tf.float32 and t.shape.rank > 1 and not
          (num_divisible(t.shape.dims, 128) >= 1 and
           num_divisible(t.shape.dims, 8) >= 2)):
      x = tf.reshape(t, [-1])
      return TPUEncodedF32(x, t.shape)
    else:
      return t

  return tf.nest.map_structure(visit, ts)


def tpu_decode(ts, structure=None):
  """Decodes a nest of Tensors encoded with tpu_encode.

  Args:
    ts: A nest of Tensors or TPUEncodedUInt8 composite tensors.
    structure: If not None, a nest of Tensors or TPUEncodedUInt8 composite
      tensors (possibly within PerReplica's) that are only used to recreate the
      structure of `ts` which then should be a list without composite tensors.

  Returns:
    A nest of decoded tensors packed as `structure` if available, otherwise
    packed as `ts`.
  """
  def visit(t, s):  
    s = s.values[0] if isinstance(s, values_lib.PerReplica) else s
    if isinstance(s, TPUEncodedUInt8):
      x = t.encoded if isinstance(t, TPUEncodedUInt8) else t
      x = tf.reshape(x, [-1, 32, 1])
      x = tf.broadcast_to(x, x.shape[:-1] + [4])
      x = tf.reshape(x, [-1, 128])
      x = tf.bitwise.bitwise_and(x, [0xFF, 0xFF00, 0xFF0000, 0xFF000000] * 32)
      x = tf.bitwise.right_shift(x, [0, 8, 16, 24] * 32)
      rank = s.original_shape.rank
      perm = [rank - 1] + list(range(rank - 1))
      inverted_shape = np.array(s.original_shape)[np.argsort(perm)]
      x = tf.reshape(x, inverted_shape)
      x = tf.transpose(x, perm)
      return x
    elif isinstance(s, TPUEncodedF32):
      x = t.encoded if isinstance(t, TPUEncodedF32) else t
      x = tf.reshape(x, s.original_shape)
      return x
    else:
      return t

  return tf.nest.map_structure(visit, ts, structure or ts)


def split_structure(structure, prefix_length, axis=0):
  """Splits in two a tf.nest structure of tensors along the first axis."""
  flattened = tf.nest.flatten(structure)
  split = [tf.split(x, [prefix_length, tf.shape(x)[axis] - prefix_length],
                    axis=axis)
           for x in flattened]
  flattened_prefix = [pair[0] for pair in split]
  flattened_suffix = [pair[1] for pair in split]
  return (tf.nest.pack_sequence_as(structure, flattened_prefix),
          tf.nest.pack_sequence_as(structure, flattened_suffix))


class nullcontext(object):  

  def __init__(self, *args, **kwds):
    del args  # unused
    del kwds  # unused

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass


def tensor_spec_from_gym_space(space, name):
  """Get a TensorSpec from a gym spec."""
  if space.shape is not None:
    return tf.TensorSpec(space.shape, space.dtype, name)
  if not isinstance(space, gym.spaces.Tuple):
    raise ValueError(
        'Space \'{}\' is not a tuple: unknown shape.'.format(space))
  num_elements = 0
  for s in space:
    if len(s.shape) != 1:
      raise ValueError(
          'Only 1 dimension subspaces are handled for tuple spaces: {}'.format(
              space))
    num_elements += s.shape[0]
  return tf.TensorSpec((num_elements,), tf.float32, name)


def validate_learner_config(config, num_hosts=1):
  """Shared part of learner config validation."""
  assert config.num_envs > 0
  assert config.env_batch_size > 0
  if config.inference_batch_size == -1:
    config.inference_batch_size = max(config.env_batch_size,
                                      config.num_envs // (2 * num_hosts))
  assert config.inference_batch_size > 0
  assert config.inference_batch_size % config.env_batch_size == 0, (
      'Learner-side batch size (=%d) must be exact multiple of the '
      'actor-side batch size (=%d).' %
      (config.inference_batch_size, config.env_batch_size))
  assert config.num_envs >= config.inference_batch_size * num_hosts, (
      'Inference batch size is bigger than the number of environments.')


def get_non_dying_envs(envs_needing_reset, reset_mask, env_ids):
  """Returns which transitions are valid or generated before an env. restarted.

  Args:
    envs_needing_reset: <int32>[num_envs_needing_reset] tensor with the IDs
      of the environments that need a reset.
    reset_mask: <bool>[inference_batch_size] tensor that contains True for
      transitions that triggered an environment reset (i.e. transition whose
      run_id does not match the previously store run_id for the corresponding
      environment).
    env_ids: <int32>[inference_batch_size] tensor of environment ID for each
      transition in the inference batch.

  Returns:
    A pair:
      - <bool>[inference_batch_size] tensor, True when the transition comes from
        a non-dying actor. False for the transitions generated by an environment
        before the transition that triggered a reset. This will typically be the
        last generated transitions before an environment restarts.
      - <int32>[num_nondying_envs] tensor, IDs of the envs that are not dying.
  """
  # <bool>[inference_batch_size] with True for all transitions coming from
  # environments that need a reset. Contrary to 'reset_mask' this covers *all*
  # transitions from the environments that have one transition that triggered
  # a reset, while 'reset_mask' only contains True for the transitions that
  # triggered a reset.
  envs_needing_reset_mask = tf.reduce_any(
      tf.equal(env_ids, tf.expand_dims(envs_needing_reset, -1)),
      axis=0)
  dying_envs_mask = tf.logical_and(
      envs_needing_reset_mask,
      tf.logical_not(reset_mask))
  num_dying_envs = tf.reduce_sum(tf.cast(dying_envs_mask, tf.int32))
  if tf.not_equal(num_dying_envs, 0):
    tf.print('Found', num_dying_envs, 'transitions from dying environments. '
             'Dying environment IDs:',
             tf.boolean_mask(env_ids, dying_envs_mask),
             'Dying environments mask:', dying_envs_mask)
  nondying_envs_mask = tf.logical_not(dying_envs_mask)
  nondying_env_ids = tf.boolean_mask(env_ids, nondying_envs_mask)
  unique_nondying_env_ids, _, unique_nondying_env_ids_count = (
      tf.unique_with_counts(nondying_env_ids))
  # If this fires, a single inference batch contains at least two transitions
  # with' the same env_id, even after filtering transitions from dying actors.
  # This can mean that an actor restarted twice while the same inference batch
  # was being filled.
  tf.debugging.Assert(
      tf.equal(tf.shape(nondying_env_ids)[0],
               tf.shape(unique_nondying_env_ids)[0]),
      data=[
          tf.gather(unique_nondying_env_ids,
                    tf.where(unique_nondying_env_ids_count >= 2)[: 0]),
          nondying_env_ids],
      summarize=4096)
  return nondying_envs_mask, nondying_env_ids


def config_from_flags():
  """Generates training config from flags.

  Returns:
    Generated training config.
  """
  config = {}
  for key in FLAGS.__flags.keys():  
    config[key] = FLAGS[key].value
  return config


def serialize_config(config):
  """Serializes training config, so that it can be send over SEED's GRPC.

  Args:
    config: config to serialize.

  Returns:
    Tensor representing serialized training config.
  """
  if isinstance(config, flags._flagvalues.FlagValues):  
    skip_keys = {'run_mode'}
    output = {}
    for key in FLAGS.__flags.keys():  
      if FLAGS[key].value != FLAGS[key].default and key not in skip_keys:
        output[key] = FLAGS[key].value
    return tf.constant(pickle.dumps(output))
  return tf.constant(pickle.dumps(config))


def update_config(current_config, client):
  """Updates current config with information from the Learner.

  Args:
    current_config: config to update.
    client: Learner's client object used to retrieve updated config.
  """
  try:
    update = client.get_config()
  except AttributeError:
    # Getting configuration is not supported by the Learner.
    return
  update = pickle.loads(update.numpy())
  if isinstance(update, dict):  
    for key, value in update:
      current_config[key] = value
  else:
    current_config = update
