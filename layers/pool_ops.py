# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
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

def maxmin_sort_2d(x, kernel=[3,3], stride=[2,2]):
    """Sort x with a spatial kernel (and stride), returns a new tensor with
    kx*ky as last sorted dimension"""
    n, h, w, c = x.shape
    grid = []
    for i in range(0, h+1-kernel[0], stride[0]):
        grid.append([])
        for j in range(0, w+1-kernel[1], stride[1]):
            grid[-1].append(x[:,i:i+kernel[0], j:j+kernel[1],:])
    tensor = tf.stack([tf.stack(row, axis=1) for row in grid], axis=1) # NHWKKC
    return maxmin_sort(tuple(tensor[:,:,:,i,j,:]
                             for i in range(kernel[0])
                             for j in range(kernel[1])))

if __name__ == "__main__":
    #%% Test for spatial sorting
    x = tf.constant(np.linspace(0,1,36))
    x = tf.reshape(x, [1, 6, 6, 1])
    sorted = maxmin_sort_2d(x, kernel=[3,3],stride=[2,2])
    with tf.Session() as sess:
        x = sess.run(sorted)
        a = x[...,:-1]
        b = x[...,1:]
        print((a>b).all())

    #%% Test for sorting over last dimension
    x = tf.random_normal((1000,9))
    sorted = maxmin_sort(x)
    with tf.Session() as sess:
        x = sess.run(sorted)
        a = x[...,:-1]
        b = x[...,1:]
        print((a>b).all())
