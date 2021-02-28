import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

import os
import pathlib

# Load feature vector extractor into KerasLayer
r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
# r50x1_loc = "embedding_models/bit_m-r50x1_1" # location of embedding model in azure data store


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# NOTE: this was edited slightly to fix issues when running locally. model itself unchanged.
def build_model(input_shape, num_classes, feat_vec_embedding=None):
  if feat_vec_embedding is not None:
    feat_vec_layer = feat_vec_embedding
  else:
    feat_vec_layer = hub.KerasLayer(r50x1_loc, name='feat_vec_embedding', trainable=False)

  input_shape = (input_shape[0], input_shape[1], 3)
  input_image = Input(shape=input_shape, name='input_image')
  rescaled_image = Rescaling(1./255, name='normalize')(input_image)
  image_embedding = feat_vec_layer(rescaled_image)

  output = Dense(num_classes, activation='softmax', name='output')(image_embedding)

  rgb_model = tf.keras.Model(inputs=[input_image], outputs=[output])

  return rgb_model

class AzureLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, run, metrics_to_log=['loss', 'accuracy']):
        super().__init__()
        self.run = run
        self.metrics_to_log = metrics_to_log

    def on_epoch_end(self, epoch, logs={}):
        print(logs)

        print('\nlogging... ', end='')
        for metric in self.metrics_to_log:
          print(f'{metric}: {logs[metric]}; ', end='')
          self.run.log(metric, np.float(logs[metric]))
