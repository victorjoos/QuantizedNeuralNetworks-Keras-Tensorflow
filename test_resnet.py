import os
import argparse
import re
from utils.load_data import load_dataset
from models.model_factory import build_model

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

def get_netw_type_wbits_abits(name):
    regex = re.compile(r'(.*)_(.*)b_(.*)b')
    m = regex.match(name)
    return m.group(1), int(m.group(2)), int(m.group(3))


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("size")
args = parser.parse_args()
size = int(args.size)
all_poss = [
    "bnn_8b_4b",
    "float_4b_4b",
    "full-bnn_4b_4b",
    "full-qnn_2b_4b",
    "full-qnn_4b_4b",
    "full-qnn_8b_4b",
    "full-tnn_4b_4b",
    "qbnn_2b_4b",
    "qbnn_4b_4b",
    "qbnn_8b_4b",
    "qnn_4b_4b",
    "qtnn_2b_4b",
    "qtnn_4b_4b",
    "qtnn_8b_4b",
    "tnn_8b_4b"
]

for np in all_poss:
    type, wbits, abits = get_netw_type_wbits_abits(np)
    cf = {
        "architecture": "RESNET",
        "dim": 32,
        "channels": 3,
        "classes": 10,
        "nres": size,
        "kernel_initializer": 'he_normal',
        "kernel_regularizer": 1e-4,
        "dataset": "CIFAR-10",
        "network_type": type,
        "wbits": wbits,
        "abits": abits
    }
    cf = obj(cf)
    model = build_model(cf)

    wname = "weights/RESNET_CIFAR-10_"+np+"_"+str(size)+".hdf5"
    model.load_weights(wname)

    train_data, val_data, test_data = load_dataset("CIFAR-10", cf)
    score = model.evaluate(test_data.X, test_data.y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
