import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import (Concatenate, Convolution2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, MaxPooling2D)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# feature vector extractor location
r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
# r50x1_loc = "embedding_models/bit_m-r50x1_1" # location of embedding model in azure data store

def build_model(input_shape, num_classes, model_name='rgbd_model'):
    # Load feature vector extractor into KerasLayer
    feat_vec_layer = hub.KerasLayer(r50x1_loc, name='feat_vec_embedding', trainable=False)

    rgb_input = Input(shape=(input_shape[0], input_shape[1], 3), name='rgb_input')
    depth_input = Input(shape=(input_shape[0], input_shape[1], 1), name='depth_input')
    #depth_input_ = tf.expand_dims(depth_input, axis=-1, name='expand_dims') # TODO

    rescaled_rgb = Rescaling(1./255, name='normalize')(rgb_input)
    rgb_embedding = feat_vec_layer(rescaled_rgb)

    depth_conv1 = Convolution2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv1')(depth_input)
    depth_maxpooling1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='depth_maxpooling1')(depth_conv1)

    depth_conv2 = Convolution2D(filters=64, kernel_size=(5,5), strides=(3, 3), padding='same', activation='relu', name='depth_conv2')(depth_maxpooling1)
    depth_maxpooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='depth_maxpooling2')(depth_conv2)

    depth_conv3 = Convolution2D(filters=64, kernel_size=(7,7), strides=(3, 3), padding='same', activation='relu', name='depth_conv3')(depth_maxpooling2)
    flat_depth = Flatten()(depth_conv3)

    rgb_depth_concat = Concatenate()([rgb_embedding, flat_depth])

    dense1 = Dense(units=128, activation='relu', name='dense1')(rgb_depth_concat)
    dense1_dropout = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu', name='dense2')(dense1_dropout)
    dense2_dropout = Dropout(0.5)(dense2)
    output = Dense(units=num_classes, activation='softmax', name='output')(dense2_dropout)

    rgbd_model = tf.keras.Model(inputs=[rgb_input, depth_input], outputs=[output], name=model_name)

    return rgbd_model

class AzureLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, run, metrics_to_log=['loss', 'accuracy']):
        super().__init__()
        self.run = run
        self.metrics_to_log = metrics_to_log

    def on_epoch_end(self, epoch, logs ={}):
        print(logs)

        print('\nlogging... ', end='')
        for metric in self.metrics_to_log:
          print(f'{metric}: {logs[metric]}; ', end='')
          self.run.log(metric, np.float(logs[metric]))
