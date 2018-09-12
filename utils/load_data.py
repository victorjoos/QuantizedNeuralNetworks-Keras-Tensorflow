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

def split_train(train_set, size):
    train = []
    valid = []
    train.append(train_set[0][0:size])
    train.append(train_set[1][0:size])
    valid.append(train_set[0][size:])
    valid.append(train_set[1][size:])
    return Dataset(train), Dataset(valid)

class Dataset:
    def __init__(self, dset):
        self.X = dset[0] / 255
        self.y = dset[1]


def load_dataset(dataset, cf):
    if (dataset == "CIFAR-10"):

        print('Loading CIFAR-10 dataset...')


        train_set_size = 45000
        train_init, test_init = cifar10.load_data()
        train_set, valid_set = split_train(train_init, train_set_size)
        test_set = Dataset(test_init)


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


    elif (dataset == "MNIST" or dataset == "FASHION"):

        print('Loading MNIST dataset...')

        train_set_size = 50000
        if dataset == "MNIST":
            train_init, test_init = mnist.load_data()
        elif dataset == "FASHION":
            train_init, test_init = fashion_mnist.load_data()
        train_set, valid_set = split_train(train_init, train_set_size)
        test_set = Dataset(test_init)
        # train_set = MNIST(which_set="train", start=0, stop=train_set_size)
        # valid_set = MNIST(which_set="train", start=train_set_size, stop=60000)
        # test_set = MNIST(which_set="test")

        train_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., train_set.X), 1.), (-1, 1, 28, 28)),(0,2,3,1))
        valid_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., valid_set.X), 1.), (-1, 1,  28, 28)),(0,2,3,1))
        test_set.X = np.transpose(np.reshape(np.subtract(np.multiply(2. / 255., test_set.X), 1.), (-1, 1,  28, 28)),(0,2,3,1))
        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)

        # Onehot the targets
        train_set.y = np.float32(np.eye(10)[train_set.y])
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])

        # for hinge loss
        train_set.y = 2 * train_set.y - 1.
        valid_set.y = 2 * valid_set.y - 1.
        test_set.y = 2 * test_set.y - 1.
        # enlarge train data set by mirrroring
        x_train_flip = train_set.X[:, :, ::-1, :]
        y_train_flip = train_set.y
        train_set.X = np.concatenate((train_set.X, x_train_flip), axis=0)
        train_set.y = np.concatenate((train_set.y, y_train_flip), axis=0)


    else:
        print("wrong dataset given")

    return train_set, valid_set, test_set

if __name__ == "__main__":
    value = load_dataset("MNIST")
