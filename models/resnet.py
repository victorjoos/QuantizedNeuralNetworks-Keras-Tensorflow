"""
Taken from https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
"""

from __future__ import print_function
import keras
from keras.layers import BatchNormalization, Lambda
from keras.layers import AveragePooling2D, Input, Flatten, ZeroPadding2D
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os

def ResNet18(Conv2D, Activation, Dense, cf):

    input_shape = (cf.dim,cf.dim,cf.channels)
    classes = cf.classes
    n = cf.nres
    depth = n * 6 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, 1)


    def resnet_layer(inputs,
                     num_filters=16,
                     kernel_size=3,
                     strides=1,
                     batch_normalization=True,
                     activation="relu",
                     conv_first=True):
        """2D Convolution-Batch Normalization-Activation stack builder

        # Arguments
            inputs (tensor): input tensor from input image or previous layer
            num_filters (int): Conv2D number of filters
            kernel_size (int): Conv2D square kernel dimensions
            strides (int): Conv2D square stride dimensions
            activation (string): activation name
            batch_normalization (bool): whether to include batch normalization
            conv_first (bool): conv-bn-activation (True) or
                bn-activation-conv (False)

        # Returns
            x (tensor): tensor as input to the next layer
        """
        conv = Conv2D(filters=num_filters*cf.pfilt,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer=cf.kernel_initializer,
                      kernel_regularizer=l2(cf.kernel_regularizer),
                      use_bias=False
                      )

        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation()(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation()(x)
            x = conv(x)
        return x

    def resnet_v1(input_shape, depth):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        if cf.dataset == "MNIST" or cf.dataset == "FASHION":
            inputs_ = ZeroPadding2D(padding=(2,2))(inputs)
        else:
            inputs_ = inputs
        x = resnet_layer(inputs=inputs_)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Lambda(lambda _x: _x * 0.5)(x)
                x = Activation()(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes=classes,
                        activation='softmax',
                        kernel_initializer=cf.kernel_initializer,
                        use_bias=False
                        )(y)
        # outputs = BatchNormalization(momentum=0.1, epsilon=1e-4)(outputs)
        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model


    return resnet_v1(input_shape=input_shape, depth=depth)
