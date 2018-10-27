# -*- coding: utf-8 -*-
"""Normalization layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.layers import interfaces
from utils.new_bn import BatchNormalization as bn2
import tensorflow as tf
import math

def my_bn(x, mean, var, beta, gamma, epsilon):
    def round_to_n(tt):
        # tt = tf.scalar_mul(1/32, tf.round(tf.scalar_mul(32, tt)))
        return tt
    def get_pow_round(tt):
        tt_sign = tf.round(tf.div(tt, tf.abs(tt)))
        tt = tf.abs(tt)
        tt = tf.round(tf.scalar_mul(1/math.log(2) , tf.log(tt)))
        tt = tf.exp(tf.scalar_mul(math.log(2), tt))
        tt = tf.multiply(tt_sign, tt)
        return tt

    beta = zeros_like(mean) if beta is None else beta
    gamma = zeros_like(mean) if gamma is None else gamma

    # y = kx + h
    # k = round_to_pow(gamma/sqrt(var+eps))
    # h = beta - mean/sqrt(var+eps)

    sve = tf.sqrt(tf.add(
                    var,
                    tf.scalar_mul(epsilon, tf.ones_like(var))
                    ))
    k = get_pow_round(tf.div(gamma, sve))
    h = round_to_n(tf.subtract(
            beta,
            tf.div(tf.multiply(mean, gamma), sve)
        ))
    return tf.add(tf.multiply(k, x), h)


class MySBN(BatchNormalization):

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 **kwargs):
        super(MySBN, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None

                return my_bn(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return my_bn(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    epsilon=self.epsilon)

        if training in {0, False}:
            # If the learning phase is *static* and set to inference:
            return normalize_inference()
        elif training is None:
            # If it's undefined then if trainable tensor is on respect learning phase else set to false
            training = K.switch(self._trainable_tensor, K.cast(K.learning_phase(), 'float32'),
                                K.constant(0, dtype='float32'))
            training._uses_learning_phase = True

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)
