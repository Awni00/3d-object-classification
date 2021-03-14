import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import (Concatenate, Convolution2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, MaxPooling2D)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

import hha_model

# feature vector extractor location
r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
# r50x1_loc = "embedding_models/bit_m-r50x1_1" # location of embedding model in azure data store


def build_hha_feature_extractor(weights=None, trainable=True):

    # build model and load trained weights
    hha_mdl = hha_model.build_model((200,200), 51)
    if weights: hha_mdl.load_weights(weights)

    # get feature extractor layers from model
    feat_vec_layers = hha_mdl.layers[:-5]

    # set trainability
    for layer in feat_vec_layers:
        layer.trainable = trainable

    # create feature vector extractor as sequential
    hha_feat_extractor = tf.keras.Sequential(layers=feat_vec_layers, name='hha_feat_extractor')

    return hha_feat_extractor

# NOTE: architecture idea: seap information from other stream while still focusing on one at a time, before final concatenation
# TODO: object segmentation model
# TODO: analyze model performance using saliency maps/class activation maps (validate use of depth info)
def build_rgb_hha_model(input_shape, num_classes, hha_feat_vec_embedding, rgb_feat_vec_embedding=None, model_name='rgb-hha_model'):
    """builds rgb-hha model using feature extractors for both rgb and hha images.

    Args:
        input_shape ((int, int)): tuple of resolution of images.
        num_classes (int): number of classes in dataset
        hha_feat_vec_embedding (keras module): feature vector extractor for hha images.
        rgb_feat_vec_embedding (keras module, optional): feature vector extractor for rgb images. Defaults to None.
        model_name (str, optional): name of model. Defaults to 'rgb-hha_model'.

    Returns:
        keras functional model: rgb-hha model
    """

    # Load feature vector extractor into KerasLayer
    if rgb_feat_vec_embedding is not None:
        rgb_feat_vec_layer = rgb_feat_vec_embedding
    else:
        r50x1_loc = "https://tfhub.dev/google/bit/m-r50x1/1"
        #r50x1_loc = "../models/bit_m-r50x1_1" # NOTE location of embedding model in azure data store
        rgb_feat_vec_layer = hub.KerasLayer(r50x1_loc, name='feat_vec_embedding', trainable=False)

    reg = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01)

    rgb_input = Input(shape=(input_shape[0], input_shape[1], 3), name='rgb_input')
    hha_input = Input(shape=(input_shape[0], input_shape[1], 3), name='hha_input')

    rescaled_rgb = Rescaling(1./255, name='normalize_rgb')(rgb_input)

    rgb_embedding = rgb_feat_vec_layer(rescaled_rgb)
    hha_embedding = hha_feat_vec_embedding(hha_input)

    rgb_hha_concat = Concatenate()([rgb_embedding, hha_embedding])

    dense1 = Dense(units=128, activation='relu', kernel_regularizer=reg, name='dense1')(rgb_hha_concat)
    dense1_dropout = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu', kernel_regularizer=reg, name='dense2')(dense1_dropout)
    dense2_dropout = Dropout(0.5)(dense2)
    output = Dense(units=num_classes, activation='softmax', kernel_regularizer=reg, name='output')(dense2_dropout)

    rgb_hha_model = tf.keras.Model(inputs=[rgb_input, hha_input], outputs=[output], name=model_name)

    return rgb_hha_model

