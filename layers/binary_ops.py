# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
from keras.layers import ReLU
import tensorflow as tf


def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return round_through(_hard_sigmoid(x))


def binary_tanh(x):
    '''Binary hard sigmoid for training binarized neural network.
     The neurons' activations binarization function
     It behaves like the sign function during forward propagation
     And like:
        hard_tanh(x) = 2 * _hard_sigmoid(x) - 1 
        clear gradient when |x| > 1 during back propagation

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    x = 2 * round_through(_hard_sigmoid(x)) - 1
    # x = round_through(_hard_sigmoid(x))
    #x = tf.Print(x,[x],summarize=10,first_n=2)
    return x

def binary_separation(x):
    x = round_through(K.clip(x, 0, 1))
    return x

def binarize(W, H=1):
    '''The weights' binarization function, 

    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    # [-H, H] -> -H or H
    Wb = H * binary_tanh(W / H)
    #Wb = tf.Print(Wb,[Wb,W],summarize=5,first_n=2)
    return Wb


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))

    
def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = binarize(W, H)
    Wa = _mean_abs(W, axis, keepdims)
    
    return Wa, Wb

from keras.layers import Layer
import tensorflow as tf
class BinaryReLU(Layer):
    def __init__(self, **kwargs):
        super(BinaryReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum =  0.9

    def call(self, inputs):
        cutoff = K.stop_gradient(tf.contrib.distributions.percentile(K.stop_gradient(inputs), 80, axis=0, keep_dims=True)) #K.mean(inputs)#
        self.threshold = K.stop_gradient(self.threshold +(cutoff-self.threshold)*self.momentum)
        thresholds = self.threshold#K.repeat(self.threshold, inputs.shape[0])
        xx = inputs - thresholds
        # xx = inputs
        # for i in range(self.size):
        #     xx[:,i] -= thresholds[i]
        print_op  = tf.print( K.sum(tf.cast(inputs[:,0]<=self.threshold[0][0], tf.int32)) )
        print_op2 = tf.print( K.sum(tf.cast(inputs[:,0] >self.threshold[0][0], tf.int32)) )
        print_op3 = tf.print(self.threshold)
        with tf.control_dependencies([print_op, print_op2, print_op3]):
            xx = K.relu(xx)#, threshold=self.threshold) # WATCH OUT this does inputs-threshold!!
        # print_op  = tf.print( K.sum(tf.cast(xx>0, tf.int32)) )
        # with tf.control_dependencies([print_op]):
        #     xx = xx-1+1
        
        # xx = K.clip(xx, 0, 1)
        # # make every value > 0 go to 1
        # xx = round_through(xx+0.499999)

        # ones = K.ones_like(inputs)
        # zeros = K.zeros_like(inputs)
        # xx = tf.where(inputs <= cutoff, zeros, ones)

        return xx
    
    def build(self, input_shape):
        self.threshold = K.zeros((1,input_shape[1]), dtype="float32")
        self.size = input_shape[1]
        self.built=True

    def get_config(self):
        config = {
            'threshold': self.threshold
        }
        base_config = super(BinaryReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
