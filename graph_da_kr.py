import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import defaultdict
import itertools
import sys
from os import listdir
from os.path import isfile, join
import re
import numpy as np
from big_parse import get_acc

do_show = False

def parse_file(file):
    arr = []
    arr2 = []
    with open(file) as fp:
        for line in fp:
            m = re.search(r"val_acc: 0.(\d+)", line)
            if m:
                arr.append(int(m[1])/100)
            m = re.search(r" acc: 0.(\d+)", line)
            if m:
                arr2.append(int(m[1])/100)
            m = re.search(r" acc: 1.", line)
            if m:
                arr2.append(100)
    return [arr, arr2]

def graph(y, label, style, color):
    x = np.arange(1, len(y)+1, 1)
    plt.plot(x, y,
                linestyle=style,
                # linewidth=0.5,
                color=color,
                #marker=markers[key[1]],
                #markersize=7,
                label=label)

def get_type(file):
    m1 = re.search(r"t(\d)", file)
    m2 = re.search(r"4(\d)", file)
    m3 = re.search(r"ff", file)
    if m1:
        return f"ternary-{m1[1]}bit"
    elif m2:
        return f"4bit-{m2[1]}bit"
    elif m3:
        return "float-float"
    else:
        return 'error'


def data_augmentation(da_kr):
    files = [f for f in listdir(da_kr) if isfile(join(da_kr, f))]
    dico = {}
    for file in files:
        if re.search(r"noda", file):
            y = parse_file(join(da_kr, file))
            dico[get_type(file)+"(no DA)"] = y
        elif not re.search(r"kr", file):
            y = parse_file(join(da_kr, file))
            dico[get_type(file)] = y

    plt.style.use("seaborn-darkgrid")
    f = plt.figure(figsize=(10,4))
    dc = {1:0, 2:0, 3:0}
    all_colors = ['C0', 'C1']
    all_labels = ["", " (no DA)"]
    keeys = {1:"float-float", 2:"ternary-4bit", 3:"4bit-4bit"}
    for key, value in dico.items():
        if "4bit-" in key:
            index = 3
        elif "ternary" in key:
            index = 2
        else:
            index = 1
        plt.subplot(1,3,index)
        graph(value[1], "training " + all_labels[dc[index]], "--", all_colors[dc[index]])
        graph(value[0], "validat. " + all_labels[dc[index]], "-",  all_colors[dc[index]])
        dc[index] += 1
        plt.legend(ncol=1, loc="lower right")
        if index==1:
            plt.ylabel("Accuracy [%]")
        plt.xlabel(f"$\\bf {{ {keeys[index]} }} $ -- Epoch [-]")
        plt.ylim([15, 105])
    plt.tight_layout()
    if do_show:
        plt.show()
    else:
        f.savefig(f"../fpga-thesis/img/data_augmentation.pdf", bbox_inches='tight')

def kernel_reg(da_kr):
    files = [f for f in listdir(da_kr) if isfile(join(da_kr, f))]
    dico = {}
    for file in sorted(files):
        if not re.search(r"noda", file) and not re.search(r"kr", file):
            y = parse_file(join(da_kr, file))
            dico[get_type(file)] = y
        elif re.search(r"kr_4", file):
            y = parse_file(join(da_kr, file))
            dico[get_type(file)+" (KR=1e-4)"] = y

    plt.style.use("seaborn-darkgrid")
    f = plt.figure(figsize=(10,4))
    dc = {1:0, 2:0, 3:0}
    all_colors = ['C0',
     'C1']
    all_labels = ["", ""]
    keeys = {1:"float-float", 2:"ternary-4bit", 3:"4bit-4bit"}
    for key, value in dico.items():
        if "4bit-" in key:
            index = 3
        elif "ternary" in key:
            index = 2
        else:
            index = 1
        plt.subplot(1,3,index)
        graph(value[1], key+" training " + all_labels[dc[index]], "--", all_colors[dc[index]])
        graph(value[0], key+" validat. " + all_labels[dc[index]], "-",  all_colors[dc[index]])
        dc[index] += 1
        plt.legend(ncol=1, loc="lower right")
        if index==1:
            plt.ylabel("Accuracy [%]")
        plt.xlabel(f"$\\bf {{ {keeys[index]} }} $ -- Epoch [-]")
        plt.ylim([15, 105])
    plt.tight_layout()
    if do_show:
        plt.show()
    else:
        f.savefig(f"../fpga-thesis/img/kernel_reg.pdf", bbox_inches='tight')

def kfold(kf, kf2):
    files = [f for f in listdir(kf) if isfile(join(kf, f))]
    files2 = [join(kf2, f) for f in listdir(kf2) if isfile(join(kf2, f))]
    fold_ff = []
    fold_t4 = []
    fold_44 = []
    for file in sorted(files):
        if re.search(r"ff_", file):
            fold_ff.append(get_acc(join(kf, file)))
        elif re.search(r"t4_", file):
            fold_t4.append(get_acc(join(kf, file)))
    for file in sorted(files2):
        fold_44.append(get_acc(file))
    data = [fold_ff, fold_t4, fold_44]
    plt.style.use('seaborn-darkgrid')
    f = plt.figure()
    sns.boxplot(data=data, palette="vlag")
    sns.swarmplot(data=data, color="0.3", size=2)
    # plt.xlabel("Threshold")
    plt.ylabel("Accuracy [%]")
    ticks = ["float-float", "ternary-4bit", "4bit-4bit"]
    plt.xticks(ticks=[i for i in range(0, len(ticks))], labels=ticks)
    if do_show:
        plt.show()
    else:
        f.savefig(f"../fpga-thesis/img/kfold.pdf", bbox_inches='tight')

def diff_lr(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    dico = {}
    for file in sorted(files):
        y = parse_file(join(folder, file))
        dico[file] = y

    plt.style.use("seaborn-darkgrid")
    f = plt.figure()
    all_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    index = 0
    for key, value in dico.items():
        graph(value[0], key, "-", all_colors[index])
        plt.legend(ncol=1, loc="lower right")
        if index==0:
            plt.ylabel("Validation Accuracy [%]")
            plt.xlabel("Epoch [-]")
            plt.ylim([5, 95])
        index += 1
    plt.tight_layout()
    if do_show:
        plt.show()
    else:
        f.savefig(f"../fpga-thesis/img/diff_lr.pdf", bbox_inches='tight')


def parse(da_kr):#, kf, kf2):
    # data_augmentation(da_kr)
    # kernel_reg(da_kr)
    # kfold(kf, kf2)
    diff_lr(da_kr)













if __name__ == "__main__":
    # parse(sys.argv[1], sys.argv[2], sys.argv[3])
    parse(sys.argv[1])
