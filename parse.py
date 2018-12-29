import os
import re
from math import sqrt, pow
from glob import glob
# import matplotlib.pyplot as plt
import random
from collections import defaultdict, Counter
from test_resnet import build_with, convert_netw_type
import keras
"""from layers.quantized_layers import QuantizedConv2D,QuantizedDense
from layers.quantized_ops import quantized_relu as quantize_op
from layers.binary_layers import BinaryConv2D, BinaryDense, Clip
from layers.binary_ops import binary_tanh
from layers.ternary_layers import TernaryConv2D, TernaryDense
from layers.ternary_ops import ternary_tanh
from keras.regularizers import l2


custom = {
    'TernaryConv2D': TernaryConv2D,
    'Clip': Clip,
    'TernaryDense': TernaryDense,
    'BinaryConv2D': BinaryConv2D,
    'QuantizedConv2D': QuantizedConv2D,
    'QuantizedDense': QuantizedDense,
    'BinaryDense': BinaryDense,
    'quantize_op': quantize_op,
    'binary_tanh': binary_tanh,
    'ternary_tanh': ternary_tanh,
    'l2': l2
}"""

models = {
    'ff': (32, 32),
    'bb': (1, 1),
    'b2': (1, 2),
    'b4': (1, 4),
    'b8': (1, 8),
    'bf': (1, 32),
    'tt': (2, 2),
    't2': (2, 2),
    't4': (2, 4),
    't8': (2, 8),
    'tf': (2, 32),
    '42': (4, 2),
    '44': (4, 4),
    '48': (4, 8),
    '4f': (4, 32),
}

def get_acc(filename):
    with open(filename) as fp:
        for line in fp:
            match = re.match(r'Test accuracy2: (0.\d*)', line)
            if match is not None:
                return float(match.groups()[0])*100
def get_acts(filename):
    """Gets activations from file."""
    with open(filename) as input_file:
        activations = 0
        for line in input_file:
            match = re.match(r'^activation.*\(None, (\d*), (\d*), (\d*)\)', line)
            if match is not None:
                gr = match.groups()
                a = int(gr[0])
                b = int(gr[1])
                c = int(gr[2])
                activations += a*b*c

        # print("total acts: ", activations)
        return activations


def get_fmap_size():
    return 32*32*16


def get_weights(filename):
    """Gets weights from file."""
    with open(filename) as input_file:
        for line in input_file:
            match = re.match(r'Total params: ([\d,]*)', line)
            if match is not None:
                return int(match.group(1).replace(',',''))


def get_input_size(filename):
    """Gets input size and # of channels."""
    with open(filename) as input_file:
        for line in input_file:
            match = re.match(r'^input.*\(None, (\d*), (\d*), (\d*)\)', line)
            if match is not None:
                x = int(match.group(1))
                c = int(match.group(3))
                return x,c


def get_Edram(Ed, s_in, c_in, Q, f_r, w_r):
    """Computes theoretical dram energy

    :param Ed: energy per *intQ* access
    :param s_in: image dimension (1 side)
    :param c_in: channel #
    :param Q: quantization
    :param f_r: # of words refetched fmap
    :param w_r: # of words refetched weights
    :return: Edram
    """
    # 8 is the number of bits of the input image
    return Ed * (s_in*s_in*c_in*8/Q + 2*f_r+w_r)


def get_Ehw(Emac, Nc, Ns, As, p):
    """Computes theoretical hw energy usage.

    :param Emac: Energy of one MAC operation
    :param Nc: Network complexity (size of weights ?)
    :param Ns: Model size
    :param As: tot # of activations
    :param p: ???
    :return: Ehw
    """
    Ec = Emac * (Nc + 3 * As)
    Ew = (2*Emac) * Ns + Emac * Nc / sqrt(p)
    Ea = 4 * Emac * As + Emac * Nc / sqrt(p)
    return Ec + Ew + Ea


def get_theoretical_Es(Q):
    """Computes theoretical estimate for power usage.

    :param Q: quantization
    :return: Emac, Ed, p
    """
    return 3.6*10**-3*Q/16, 0.4608*(Q/16), 64*16/Q


def get_memory():
    """See https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/hb/cyclone-v/cv_51001.pdf."""
    return 13917*10**3  # maybe should be * 1024


def rreplace(s, old, new, occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def compute_estimate(wbit, abit, log, wname, type, nres):
    print(wname, wbit, abit)
    accuracy = get_acc(log)#build_with(wname, type, nres)
    Q = wbit
    As = get_acts(log)
    Ns = get_weights(log)
    Nc = Ns # error here
    s_in, c_in = get_input_size(log)

    mem = get_memory()
    fmap = get_fmap_size()

    if mem//2 - fmap < 0:
        fr = fmap - mem//2
    else:
        fr = 0
    if mem//2 - Ns*Q < 0:
        wr = Ns*Q - mem//2
    else:
        wr = 0
    Emac, Ed, p = get_theoretical_Es(Q)
    Edram = get_Edram(Ed, s_in, c_in, Q, fr, wr)
    Ehw = get_Ehw(Emac, Nc, Ns, As, p)
    print("Energy [uJ] : ", Edram, Ehw)
    outfile.write("{}, {}, {}, {}\n".format(type, nres, accuracy, Edram + Ehw))
    return accuracy, Edram + Ehw


def parse_models(dir, accs, energy):
    logs_dir = os.path.join(dir, "logs/")
    logs = os.path.join(logs_dir, "*.out")

    for log in glob(logs):
        type = log[-6:-4]
        wname = os.path.join(dir, "RESNET_CIFAR-10_"+convert_netw_type(type)+"_*.hdf5")
        for x in glob(wname):
            wname = x
            match = re.match(r'(.*)b_(.*)b_(\d*).hdf5', wname)
            nres = int(match.group(3))
            print(nres)
            break
        if os.path.exists(wname):
            acc, ener = compute_estimate(*models[type], log, wname, type, nres)
            accs[type].append(acc)
            energy[type].append(ener)
        else:
            print("log file has no weights !")

def parse_dir(dir_list):
    accs = defaultdict(list)
    energy = defaultdict(list)
    for dir in dir_list:
        parse_models(dir, accs, energy)

    # for key, value in accs.items():
    #     plt.semilogy(value, energy[key])
    #
    # plt.show()


import sys
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dir_list = sys.argv[1:len(sys.argv)]
    outfile = open("results.csv", 'w')
    parse_dir(dir_list)
