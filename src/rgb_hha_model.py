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
def build_model(input_shape, num_classes, feat_vec_embedding=None, model_name='rgb_hha_model'):
    # Load feature vector extractor into KerasLayer
    if feat_vec_embedding is not None:
        feat_vec_layer = feat_vec_embedding
    else:
        r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
        #r50x1_loc = "../models/bit_m-r50x1_1" # NOTE location of embedding model in azure data store
        feat_vec_layer = hub.KerasLayer(r50x1_loc, name='feat_vec_embedding', trainable=False)

    reg = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01)

    rgb_input = Input(shape=(input_shape[0], input_shape[1], 3), name='rgb_input')
    hha_input = Input(shape=(input_shape[0], input_shape[1], 3), name='hha_input')

    rescaled_rgb = Rescaling(1./255, name='normalize_rgb')(rgb_input)
    rescaled_hha = Rescaling(1./255, name='normalize_hha')(hha_input)

    rgb_embedding = feat_vec_layer(rescaled_rgb)
    hha_embedding = feat_vec_layer(rescaled_hha)

    rgb_hha_concat = Concatenate()([rgb_embedding, hha_embedding])

    dense1 = Dense(units=64, activation='relu', kernel_regularizer=reg, name='dense1')(rgb_hha_concat)
    dense1_dropout = Dropout(0.5)(dense1)
    output = Dense(units=num_classes, activation='softmax', kernel_regularizer=reg, name='output')(dense1_dropout)

    rgb_hha_model = tf.keras.Model(inputs=[rgb_input, hha_input], outputs=[output], name=model_name)

    return rgb_hha_model

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
