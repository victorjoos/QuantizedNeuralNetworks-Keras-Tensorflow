# -*- coding: utf-8 -*-
'''ResNet18 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings
from keras.models import Model
from keras.layers import Input
from keras import layers
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
import keras.backend as K



def ResNet18(Conv2D,
             Activation,
             Dense,
             input_shape,
             classes):

    def identity_block(input_tensor, filters, stage, block):

        filters1, filters2 = filters
        bn_axis = 3
        kernel_size = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, kernel_size, name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation()(x)

        x = Conv2D(filters2, kernel_size, name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

        x = layers.add([x, input_tensor])
        x = Activation()(x)
        return x


    def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):

        filters1, filters2 = filters
        bn_axis = 3
        kernel_size = 3

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, kernel_size, strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation()(x)

        x = Conv2D(filters2, kernel_size,
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

        shortcut = Conv2D(filters2, 1, strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation()(x)
        return x

    bn_axis = 3

    # x = ZeroPadding2D((3, 3))(img_input)
    img_input = Input(shape=input_shape)
    x = Conv2D(64, 3, strides=(1, 1), name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation()(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, [64, 64], stage=2, block='b')

    x = conv_block(x, [128, 128], stage=3, block='a')
    x = identity_block(x, [128, 128], stage=3, block='b')


    x = conv_block(x, [256, 256], stage=4, block='a')
    x = identity_block(x, [256, 256], stage=4, block='b')

    x = conv_block(x, [512, 512], stage=5, block='a')
    x = identity_block(x, [512, 512], stage=5, block='b')

    x = AveragePooling2D((4, 4), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes)(x) #  activation='softmax', name='classify'
    x = BatchNormalization(momentum=0.1,epsilon=0.0001)(x)

    # Create model.
    model = Model(img_input, x, name='resnet18')

    return model
