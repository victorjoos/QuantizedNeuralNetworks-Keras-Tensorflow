
import numpy as np
import tensorflow as tf

def maxmin_sort(tensor):
    """Takes a tensor (or a tuple of tensors) and returns it sorted over the last
    dimension (using min-max swaps for parallelization-friendliness)"""
    if not isinstance(tensor, tuple):
        n = tensor.shape[-1]
        tensor = tuple(tensor[...,i] for i in range(n))
    else:
        n = len(tensor)
    for i in range(n):
        offset = i & 1
        tensor = sum([(tf.maximum(tensor[j], tensor[j+1]),
                       tf.minimum(tensor[j], tensor[j+1]))
                      if j <= n-2 else (tensor[j],)
                      for j in range(offset,n,2)], tensor[:offset])
    return tf.stack(tensor, axis=-1)

def maxmin_sort_2d(x, pool_size=(2,2), stride=None):
    """Sort x with a spatial kernel (and stride), returns a new tensor with
    kx*ky as last sorted dimension"""
    n, h, w, c = x.shape.as_list()
    custom_stride = stride is not None
    kernel = pool_size
    if custom_stride:
        grid = []
        for i in range(0, h+1-kernel[0], stride[0]):
            grid.append([])
            for j in range(0, w+1-kernel[1], stride[1]):
                grid[-1].append(x[:,i:i+kernel[0], j:j+kernel[1],:])
        tensor = tf.stack([tf.stack(row, axis=2) for row in grid], axis=1) # NHkWkC
    else:
        tensor = tf.reshape(x, [-1, h//kernel[0], kernel[0], w//kernel[1], kernel[1], c])
    return maxmin_sort(tuple(tensor[:,:,i,:,j,:]
                             for i in range(kernel[0])
                             for j in range(kernel[1])))
#%%
from keras.models import Model
from keras.layers import Layer, Input, Lambda, Concatenate, MaxPool2D, GlobalMaxPool2D, AvgPool2D, GlobalAvgPool2D
from keras import backend as K

from keras.constraints import Constraint

class NormalizedPositiveWeights(Constraint):
    def __init__(self, normalize_weights=False):
        self.normalize_weights = normalize_weights
        
    def __call__(self, w):
        w = w * K.cast(K.greater_equal(w, 0.), K.floatx()) # Non negativity constraint
        if self.normalize_weights:
            w = w / K.sum(w, axis=-1, keepdims=True) #Normalized weights constraint
        return w


def avg_pooling_init(shape):
    weights = np.ones(shape) / shape[1]
    return weights


def max_pooling_init(shape):
    weights = np.ones(shape) * np.array([1]+[0]*(shape[1]-1))[None]
    return weights


def random(shape):
    return np.random.rand(*shape)


class OrdinalPooling2D(Layer):
    """A layer that pools any tensor of shape [N, H, W, C] into a tensor of shape [N, H/2, W/2, C]
    It pools 2x2 squares by ordering activations to obtain a b
                                                           c d
    where a >= b >= c >= d, and then summing them with trainable weights wa, wb, wc, wd.
    wa, wb, wc, wd can be specific per channel or not, and can be forced to sum together to 1 or not.
    """
    def __init__(self, pool_size=(2,2), stride=None, padding=None, normalize_weights=True, weights_per_channel=True,
                 initializer=None, softmax=False, **kwargs):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.normalize_weights = normalize_weights
        self.weights_per_channel = weights_per_channel
        self.softmax = softmax
        self.initializer = random if initializer is None else initializer
        super(OrdinalPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1] if self.weights_per_channel else 1
        self.ordinal_weights = self.add_weight(name="ordinal_weights",
                                            shape=[channels, self.pool_size[0]*self.pool_size[1]],
                                            initializer=self.initializer,
                                            constraint=NormalizedPositiveWeights(self.normalize_weights)\
                                                       if not self.softmax else None,
                                            trainable=True)
        super(OrdinalPooling2D, self).build(input_shape)

    def call(self, x):
        weights = self.ordinal_weights
        if self.softmax:
            weights = K.softmax(weights)
        elif self.normalize_weights:
            weights = weights * K.cast(K.greater_equal(weights, 0.), K.floatx())
            weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
        padding = self.padding
        if padding is not None:
            x = tf.pad(x, [[0,0],self.padding[0],self.padding[1],[0,0]], "CONSTANT")
        x = maxmin_sort_2d(x, pool_size=self.pool_size, stride=self.stride) # NhwCK
        x = x * weights[None,None,None,:,:]
        x = tf.reduce_sum(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        n, h, w, c = input_shape
        return n, h//self.pool_size[0], w//self.pool_size[1], c
