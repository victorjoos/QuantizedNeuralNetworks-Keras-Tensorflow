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
from keras.layers import BatchNormalization
import tensorflow as tf

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
            def round_to_n(tensor):
                return tensor#tf.scalar_mul(1/16, tf.round(tf.scalar_mul(16, tensor)))

            beta = zeros_like(mean) if self.beta is None else self.beta
            gamma = zeros_like(mean) if self.gamma is None else self.gamma

            return tf.add(
                        tf.multiply(
                                tf.subtract(inputs, self.moving_mean),
                                round_to_n(tf.div(
                                        gamma,
                                        tf.sqrt(tf.add(
                                                        self.moving_variance,
                                                        tf.scalar_mul(1e-3, tf.ones_like(self.moving_variance))
                                                ))
                                       ))
                              ),
                        round_to_n(beta)
                        )


        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = _fused_normalize_batch_in_training(
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
        return in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

def _fused_normalize_batch_in_training(x, gamma, beta, reduction_axes,
                                       epsilon=1e-3):
    if list(reduction_axes) == [0, 1, 2]:
        normalization_axis = 3
        tf_data_format = 'NHWC'
    else:
        normalization_axis = 1
        tf_data_format = 'NCHW'

    if gamma is None:
        gamma = tf.constant(1.0,
                            dtype=x.dtype,
                            shape=[x.get_shape()[normalization_axis]])
    if beta is None:
        beta = tf.constant(0.0,
                           dtype=x.dtype,
                           shape=[x.get_shape()[normalization_axis]])

    return tf.nn.fused_batch_norm(
        x,
        gamma,
        beta,
        epsilon=epsilon,
        data_format=tf_data_format)
def in_train_phase(x, alt, training=None):
    x = alt
    if training is None:
        training = K.learning_phase()
        uses_learning_phase = True
    else:
        uses_learning_phase = False

    if training is 1 or training is True:
        if callable(x):
            return x()
        else:
            return x

    elif training is 0 or training is False:
        if callable(alt):
            return alt()
        else:
            return alt

    # else: assume learning phase is a placeholder tensor.
    x = K.switch(training, x, alt)
    if uses_learning_phase:
        x._uses_learning_phase = True
    return x
