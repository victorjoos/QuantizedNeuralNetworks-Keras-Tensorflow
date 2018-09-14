# Copyright 2017 Bert Moons

# This file is part of QNN.

# QNN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# QNN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# The code for QNN is based on BinaryNet: https://github.com/MatthieuCourbariaux/BinaryNet

# You should have received a copy of the GNU General Public License
# along with QNN.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
# from pylearn2.datasets.cifar10 import CIFAR10
# from pylearn2.datasets.mnist import MNIST
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras import backend as K
import keras

def split_train(train_set, size, add_dim=False):
    train = []
    valid = []
    train.append(train_set[0][0:size])
    train.append(train_set[1][0:size])
    valid.append(train_set[0][size:])
    valid.append(train_set[1][size:])
    return Dataset(train, add_dim), Dataset(valid, add_dim)

class Dataset:
    def __init__(self, dset, add_dim=False):
        normX = dset[0].astype('float32') / 255
        self.X = np.reshape(normX, dset[0].shape+(1,)) if add_dim else normX
        self.y = dset[1]


def load_dataset(dataset, cf):
    if (dataset == "CIFAR-10"):

        print('Loading CIFAR-10 dataset...')


        train_set_size = 50000
        train_init, test_init = cifar10.load_data()
        train_set, valid_set = split_train(train_init, train_set_size)
        test_set = Dataset(test_init)
    elif (dataset == "MNIST" or dataset == "FASHION"):

        print('Loading MNIST dataset...')

        train_set_size = 50000
        if dataset == "MNIST":
            train_init, test_init = mnist.load_data()
        elif dataset == "FASHION":
            train_init, test_init = fashion_mnist.load_data()
        train_set, valid_set = split_train(train_init, train_set_size, True)
        test_set = Dataset(test_init, True)

    else:
        raise ValueError(str(dataset)+ " is not supported")


    # If subtract pixel mean is enabled
    subtract_pixel_mean = False
    if subtract_pixel_mean:
        x_mean = np.mean(train_set.X, axis=0)
        train_set.X -= x_mean
        test_set.X -= x_mean
        valid_set.X -= x_mean

    # flatten targets
    train_set.y = keras.utils.to_categorical(train_set.y, 10)
    valid_set.y = keras.utils.to_categorical(valid_set.y, 10)
    test_set.y = keras.utils.to_categorical(test_set.y, 10)

    # for hinge loss
    if cf.architecture=="VGG":
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.

    print(train_set.X.shape)
    valid_set = test_set
    return train_set, valid_set, test_set

if __name__ == "__main__":
    value = load_dataset("MNIST")
