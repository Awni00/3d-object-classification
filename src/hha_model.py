import tensorflow as tf
import tensorflow_hub as hub


import numpy as np

import os
import pathlib

# Load feature vector extractor into KerasLayer
r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
# r50x1_loc = "embedding_models/bit_m-r50x1_1" # location of embedding model in azure data store


from tensorflow.keras.layers import (Concatenate, Convolution2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, MaxPooling2D)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def build_model(input_shape, num_classes, model_name='hha_model'):
    '''build hha image classification model'''

    hha_input = Input(shape=(input_shape[0], input_shape[1], 3), name='hha_input')
    rescaled_hha = Rescaling(1./255, name='normalize')(hha_input)

    reg = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01)


    hha_conv1 = Convolution2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='hha_conv1')(rescaled_hha)
    hha_maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='hha_maxpool1')(depth_conv1)

    hha_conv2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='hha_conv2')(depth_maxpool1)
    hha_maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='hha_maxpool2')(depth_conv2)

    hha_conv3 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='hha_conv3')(depth_maxpool2)
    hha_maxpool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='hha_maxpool3')(depth_conv3)

    hha_conv4 = Convolution2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='hha_conv4')(depth_maxpool3)
    hha_maxpool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='hha_maxpool4')(depth_conv4)

    hha_conv5 = Convolution2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='hha_conv5')(depth_maxpool4)
    hha_maxpool5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='hha_maxpool5')(depth_conv5)

    hha_conv6 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='hha_conv6')(depth_maxpool5)
    hha_maxpool6 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='hha_maxpool6')(depth_conv6)

    hha_feat_vec = Flatten()(depth_maxpool6)

    dense1 = Dense(units=128, activation='relu', kernel_regularizer=reg, name='dense1')(depth_feat_vec)
    dense1_dropout = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu', kernel_regularizer=reg, name='dense2')(dense1_dropout)
    dense2_dropout = Dropout(0.5)(dense2)
    output = Dense(units=num_classes, activation='softmax', name='output')(dense2_dropout)

    hha_model = tf.keras.Model(inputs=hha_input, outputs=[output], name=model_name)

    return hha_model

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
