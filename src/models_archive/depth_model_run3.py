import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import (Concatenate, Convolution2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, MaxPooling2D)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


# TODO: normalization/rescaling on depth image (IMPORTANT)
def build_model(input_shape, num_classes, model_name='depth_model'):
    '''build depth image classification model'''

    depth_input = Input(shape=(input_shape[0], input_shape[1], 1), name='depth_input')
    rescaled_depth = Rescaling(1./255, name='normalize')(depth_input)

    reg = tf.keras.regularizers.l1_l2(l1=0.0, l2=0.01)


    depth_conv1 = Convolution2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same', activation='relu', name='depth_conv1')(rescaled_depth)
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

    dense1 = Dense(units=128, activation='relu', kernel_regularizer=reg, name='dense1')(depth_feat_vec)
    dense1_dropout = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu', kernel_regularizer=reg, name='dense2')(dense1_dropout)
    dense2_dropout = Dropout(0.5)(dense2)
    output = Dense(units=num_classes, activation='softmax', name='output')(dense2_dropout)

    depth_model = tf.keras.Model(inputs=depth_input, outputs=[output], name=model_name)

    return depth_model