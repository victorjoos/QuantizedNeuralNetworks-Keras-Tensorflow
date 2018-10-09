import os
import argparse
import re
from utils.load_data import load_dataset
from models.model_factory import build_model
import sys
from keras.models import Model
import numpy as np

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, obj(b) if isinstance(b, dict) else b)

def convert_netw_type(_type):
    all_poss = {
        "bf":"bnn_8b_4b",
        "ff":"float_4b_4b",
        "bb":"full-bnn_4b_4b",
        "42":"full-qnn_2b_4b",
        "44":"full-qnn_4b_4b",
        "48":"full-qnn_8b_4b",
        "tt":"full-tnn_4b_4b",
        "b2":"qbnn_2b_4b",
        "b4":"qbnn_4b_4b",
        "b8":"qbnn_8b_4b",
        "4f":"qnn_4b_4b",
        "t2":"qtnn_2b_4b",
        "t4":"qtnn_4b_4b",
        "t8":"qtnn_8b_4b",
        "tf":"tnn_8b_4b"
    }
    return all_poss[_type]

def get_netw_type_wbits_abits(name):
    regex = re.compile(r'(.*)_(.*)b_(.*)b')
    m = regex.match(name)
    return m.group(1), int(m.group(2)), int(m.group(3))


def build_with(filename, oldtype ,nres):
    newp = convert_netw_type(oldtype)
    _type, wbits, abits = get_netw_type_wbits_abits(newp)
    print(_type, wbits, abits)
    cf = {
        "architecture": "RESNET",
        "dim": 32,
        "channels": 3,
        "classes": 10,
        "nres": nres,
        "kernel_initializer": 'he_normal',
        "kernel_regularizer": 1e-4,
        "dataset": "CIFAR-10",
        "network_type": _type,
        "wbits": wbits,
        "abits": abits
    }
    cf = obj(cf)
    model = build_model(cf)

    wname = filename
    model.load_weights(wname)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    layer_name = "dense_1"
    intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

    train_data, val_data, test_data = load_dataset("CIFAR-10", cf)
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg

    # imgplot = plt.imshow(test_data.X[0])
    # plt.show()
    # print(test_data.X[0].flatten())
    score = model.evaluate(test_data.X, test_data.y) # intermediate_layer_model.predict(np.array([test_data.X[0]]), verbose=0)
    print(score)
    """for i in range(len(score[0])):
        for j in range(len(score[0][i])):
            print("{0: .5f} ".format(score[0][i][j][0]), end='')
        print('')"""
    print(score.shape)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # print(score)
    return score#[1]

if __name__ == "__main__":
    build_with(sys.argv[1], "ff", 3)
