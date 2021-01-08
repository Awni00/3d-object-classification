import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

import os
import pathlib

# Load feature vector extractor into KerasLayer
r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
# r50x1_loc = "embedding_models/bit_m-r50x1_1" # location of embedding model in azure data store
feat_vec_layer = hub.KerasLayer(r50x1_loc, name='feat_vec_embedding', trainable=False)


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def build_model(input_shape, num_classes):
  input_image = Input(shape=input_shape, name='input_image')
  rescaled_image = Rescaling(1./255, name='normalize')(input_image)
  image_embedding = feat_vec_layer(rescaled_image)
  output = Dense(num_classes, activation='softmax', name='output')(image_embedding)
  rgb_model = tf.keras.Model(inputs=[input_image], outputs=[output])

  return rgb_model

# class RGB_model(tf.keras.Model):
#   """transfer learning model using feature vector and custom new head"""

#   def __init__(self, num_classes, embedding_layer):
#     super().__init__()

#     self.num_classes = num_classes
#     self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, offset= -1)
#     self.embedding_layer = embedding_layer
#     self.predict = tf.keras.layers.Dense(num_classes)

#   def call(self, images):
#     rescaled = self.rescale(images)
#     img_embedding = self.embedding_layer(rescaled)
#     pred = self.predict(img_embedding)

#     return pred


