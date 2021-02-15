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


# NOTE: architecture idea: seap information from other stream while still focusing on one at a time, before final concatenation
# TODO: depth-only model to train feature extraction without being confused by rgb info then use weights in full rgb-d model
# TODO: object segmentation model
# TODO: analyze model performance using saliency maps/class activation maps (validate use of depth info)
# TODO: normalization/rescaling on depth image (IMPORTANT)
# TODO: pre-trained/partially trained rgb model on HHA image
def build_model(input_shape, num_classes, feat_vec_embedding=None, model_name='rgbd_model'):
    # Load feature vector extractor into KerasLayer
    if feat_vec_embedding is not None:
        feat_vec_layer = feat_vec_embedding
    else:
        r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
        #r50x1_loc = "../models/bit_m-r50x1_1" # NOTE location of embedding model in azure data store
        feat_vec_layer = hub.KerasLayer(r50x1_loc, name='feat_vec_embedding', trainable=False)

    rgb_input = Input(shape=(input_shape[0], input_shape[1], 3), name='rgb_input')
    depth_input = Input(shape=(input_shape[0], input_shape[1], 1), name='depth_input')

    rescaled_rgb = Rescaling(1./255, name='normalize')(rgb_input)
    rgb_embedding = feat_vec_layer(rescaled_rgb)

    depth_conv1 = Convolution2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv1')(depth_input)
    depth_maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='depth_maxpool1')(depth_conv1)

    depth_conv2 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv2')(depth_maxpool1)
    depth_maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='depth_maxpool2')(depth_conv2)

    depth_conv3 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv3')(depth_maxpool2)
    depth_maxpool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='depth_maxpool3')(depth_conv3)

    depth_conv4 = Convolution2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv4')(depth_maxpool3)
    depth_maxpool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='depth_maxpool4')(depth_conv4)

    depth_conv5 = Convolution2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv5')(depth_maxpool4)
    depth_maxpool5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='depth_maxpool5')(depth_conv5)

    depth_conv6 = Convolution2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv6')(depth_maxpool5)
    depth_maxpool6 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='depth_maxpool6')(depth_conv6)

    depth_feat_vec = Flatten()(depth_maxpool6)

    rgb_depth_concat = Concatenate()([rgb_embedding, depth_feat_vec])

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

        print('\nlogging... ', end='')
        for metric in self.metrics_to_log:
          print(f'{metric}: {logs[metric]}; ', end='')
          self.run.log(metric, np.float(logs[metric]))
