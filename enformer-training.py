#!/usr/bin/env python
# coding: utf-8

# Copyright 2021 DeepMind Technologies Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This colab showcases training of the Enformer model published in
# 
# **"Effective gene expression prediction from sequence by integrating long-range interactions"**
# 
# Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam, Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, David R. Kelley
# 

# ## Steps
# 
# - Setup tf.data.Dataset by directly accessing the Basenji2 data on GCS: `gs://basenji_barnyard/data`
# - Train the model for a few steps, alternating training on human and mouse data batches
# - Evaluate the model on human and mouse genomes

# ## Setup

# **Start the colab kernel with GPU**: Runtime -> Change runtime type -> GPU

# ### Install dependencies

# In[1]:


get_ipython().system('pip install dm-sonnet tqdm')


# In[15]:


# Get enformer source code
get_ipython().system('wget -q https://raw.githubusercontent.com/deepmind/deepmind-research/master/enformer/attention_module.py')
get_ipython().system('wget -q https://raw.githubusercontent.com/deepmind/deepmind-research/master/enformer/enformer.py')


# ### Import

# In[4]:


import tensorflow as tf
# Make sure the GPU is enabled 
assert tf.config.list_physical_devices('GPU'), 'Start the colab kernel with GPU: Runtime -> Change runtime type -> GPU'

# Easier debugging of OOM
get_ipython().run_line_magic('env', 'TF_ENABLE_GPU_GARBAGE_COLLECTION=false')


# In[5]:


import sonnet as snt
from tqdm import tqdm
from IPython.display import clear_output
import numpy as np
import pandas as pd
import time
import os


# In[6]:


assert snt.__version__.startswith('2.0')


# In[7]:


tf.__version__


# In[8]:


# GPU colab has T4 with 16 GiB of memory
get_ipython().system('nvidia-smi')


# ### Code

# In[37]:


import enformer


# In[41]:


# @title `get_targets(organism)`
def get_targets(organism):
  targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
  return pd.read_csv(targets_txt, sep='\t')


# In[42]:


# @title `get_dataset(organism, subset, num_threads=8)`
import glob
import json
import functools


def organism_path(organism):
  return os.path.join('gs://basenji_barnyard/data', organism)


def get_dataset(organism, subset, num_threads=8):
  metadata = get_metadata(organism)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset


def get_metadata(organism):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length
  path = os.path.join(organism_path(organism), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)

  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)

  return {'sequence': sequence,
          'target': target}


# ## Load dataset

# In[43]:


df_targets_human = get_targets('human')
df_targets_human.head()


# In[44]:


human_dataset = get_dataset('human', 'train').batch(1).repeat()
mouse_dataset = get_dataset('mouse', 'train').batch(1).repeat()
human_mouse_dataset = tf.data.Dataset.zip((human_dataset, mouse_dataset)).prefetch(2)


# In[45]:


it = iter(mouse_dataset)
example = next(it)


# In[46]:


# Example input
it = iter(human_mouse_dataset)
example = next(it)
for i in range(len(example)):
  print(['human', 'mouse'][i])
  print({k: (v.shape, v.dtype) for k,v in example[i].items()})


# ## Model training

# In[51]:


def create_step_function(model, optimizer):

  @tf.function
  def train_step(batch, head, optimizer_clip_norm_global=0.2):
    with tf.GradientTape() as tape:
      outputs = model(batch['sequence'], is_training=True)[head]
      loss = tf.reduce_mean(
          tf.keras.losses.poisson(batch['target'], outputs))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply(gradients, model.trainable_variables)

    return loss
  return train_step


# In[52]:


learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
num_warmup_steps = 5000
target_learning_rate = 0.0005

model = enformer.Enformer(channels=1536 // 4,  # Use 4x fewer channels to train faster.
                          num_heads=8,
                          num_transformer_layers=11,
                          pooling_type='max')

train_step = create_step_function(model, optimizer)


# In[59]:


# Train the model
steps_per_epoch = 20
num_epochs = 5

data_it = iter(human_mouse_dataset)
global_step = 0
for epoch_i in range(num_epochs):
  for i in tqdm(range(steps_per_epoch)):
    global_step += 1

    if global_step > 1:
      learning_rate_frac = tf.math.minimum(
          1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))      
      learning_rate.assign(target_learning_rate * learning_rate_frac)

    batch_human, batch_mouse = next(data_it)

    loss_human = train_step(batch=batch_human, head='human')
    loss_mouse = train_step(batch=batch_mouse, head='mouse')

  # End of epoch.
  print('')
  print('loss_human', loss_human.numpy(),
        'loss_mouse', loss_mouse.numpy(),
        'learning_rate', optimizer.learning_rate.numpy()
        )


# ## Evaluate

# In[60]:


# @title `PearsonR` and `R2` metrics

def _reduced_shape(shape, axis):
  if axis is None:
    return tf.TensorShape([])
  return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
  """Contains shared code for PearsonR and R2."""

  def __init__(self, reduce_axis=None, name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
      name: Metric name.
    """
    super(CorrelationStats, self).__init__(name=name)
    self._reduce_axis = reduce_axis
    self._shape = None  # Specified in _initialize.

  def _initialize(self, input_shape):
    # Remaining dimensions after reducing over self._reduce_axis.
    self._shape = _reduced_shape(input_shape, self._reduce_axis)

    weight_kwargs = dict(shape=self._shape, initializer='zeros')
    self._count = self.add_weight(name='count', **weight_kwargs)
    self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
    self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
    self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
    self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
    self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Update the metric state.

    Args:
      y_true: Multi-dimensional float tensor [batch, ...] containing the ground
        truth values.
      y_pred: float tensor with the same shape as y_true containing predicted
        values.
      sample_weight: 1D tensor aligned with y_true batch dimension specifying
        the weight of individual observations.
    """
    if self._shape is None:
      # Explicit initialization check.
      self._initialize(y_true.shape)
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    self._product_sum.assign_add(
        tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

    self._true_sum.assign_add(
        tf.reduce_sum(y_true, axis=self._reduce_axis))

    self._true_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

    self._pred_sum.assign_add(
        tf.reduce_sum(y_pred, axis=self._reduce_axis))

    self._pred_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

    self._count.assign_add(
        tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

  def result(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def reset_states(self):
    if self._shape is not None:
      tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                        for v in self.variables])


class PearsonR(CorrelationStats):
  """Pearson correlation coefficient.

  Computed as:
  ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
  """

  def __init__(self, reduce_axis=(0,), name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                   name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    pred_mean = self._pred_sum / self._count

    covariance = (self._product_sum
                  - true_mean * self._pred_sum
                  - pred_mean * self._true_sum
                  + self._count * true_mean * pred_mean)

    true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
    pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    correlation = covariance / tp_var

    return correlation


class R2(CorrelationStats):
  """R-squared  (fraction of explained variance)."""

  def __init__(self, reduce_axis=None, name='R2'):
    """R-squared metric.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(R2, self).__init__(reduce_axis=reduce_axis,
                             name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    total = self._true_squared_sum - self._count * tf.math.square(true_mean)
    residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)

    return tf.ones_like(residuals) - residuals / total


class MetricDict:
  def __init__(self, metrics):
    self._metrics = metrics

  def update_state(self, y_true, y_pred):
    for k, metric in self._metrics.items():
      metric.update_state(y_true, y_pred)

  def result(self):
    return {k: metric.result() for k, metric in self._metrics.items()}


# In[61]:


def evaluate_model(model, dataset, head, max_steps=None):
  metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
  @tf.function
  def predict(x):
    return model(x, is_training=False)[head]

  for i, batch in tqdm(enumerate(dataset)):
    if max_steps is not None and i > max_steps:
      break
    metric.update_state(batch['target'], predict(batch['sequence']))

  return metric.result()


# In[66]:


metrics_human = evaluate_model(model,
                               dataset=get_dataset('human', 'valid').batch(1).prefetch(2),
                               head='human',
                               max_steps=100)
print('')
print({k: v.numpy().mean() for k, v in metrics_human.items()})


# In[63]:


metrics_mouse = evaluate_model(model,
                               dataset=get_dataset('mouse', 'valid').batch(1).prefetch(2),
                               head='mouse',
                               max_steps=100)
print('')
print({k: v.numpy().mean() for k, v in metrics_mouse.items()})

