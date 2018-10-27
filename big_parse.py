import os
import re
from math import sqrt, pow
from glob import glob
# import matplotlib.pyplot as plt
import random
from collections import defaultdict, Counter
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
            match = re.match(r'^(?:activation|leaky_re_lu).*\(None, (\d*), (\d*), (\d*)\)', line)
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


def compute_estimate(log, ktype, nres):
    wbit, abit = models[ktype]
    accuracy = get_acc(log)
    Q = wbit#0.6*wbit+0.4*abit
    As = get_acts(log)
    Ns = get_weights(log)
    Nc = Ns
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
    return accuracy, Edram + Ehw

def compute_simple(log, ktype, nres):
    wbit, abit = models[ktype]
    accuracy = get_acc(log)
    As = get_acts(log)
    Ns = get_weights(log)
    print(Ns/As)
    # if As == 0:
    #     print(ktype, log)
    return accuracy, wbit*Ns + abit*As#*1.5


def parse_dir(direc):
    for pfi in [1, 2, 4]:
        for nres in [3, 5]:
            for key in models.keys():
                sol = None
                for subdir in [sd for sd in os.listdir(direc) if os.path.isdir(os.path.join(direc, sd))]:
                    conf = os.path.join(direc, subdir, "logs", "config.txt")
                    kfile = os.path.join(direc, subdir, "logs", key+".out")
                    with open(conf, 'r') as fd:
                        for line in fd:
                            reg = re.match(".* lr=(.*) nres=(.*) pfilt=(.*) cuda", line)
                            if not reg:
                                reg = re.match(".* lr=(.*) nres=(.*) cuda", line)
                                fpfi = 1
                            else:
                                fpfi = int(reg.group(3))
                            fnres = int(reg.group(2))
                            lr = float(reg.group(1))
                            if fnres == nres and fpfi==pfi and os.path.isfile(kfile):
                                acc, en = compute_simple(kfile, key, nres)
                                if acc and (not sol or sol[0] < acc):
                                    sol = (acc, en, lr)
                            break
                if sol:
                    outfile.write("{}, {}, {}, {}, {}\n".format(key, nres, sol[0], sol[1], sol[2]))

import sys
if __name__ == '__main__':
    outfile = open("results.csv", 'w')
    parse_dir(sys.argv[1])
