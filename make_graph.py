import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import sys

with open(sys.argv[1]) as results:
    accs = defaultdict(list)
    Es = defaultdict(list)
    for line in results:
        vals = line.split(", ")
        type = vals[0]
        acc, energy = float(vals[2]), float(vals[3])
        accs[type].append(acc)
        Es[type].append(energy)
    print(accs)
    marker = itertools.cycle(('+', '.', 'o', '*'))
    f = plt.figure()
    for key, value in accs.items():
        energy = Es[key]
        plt.semilogy(value, energy, linestyle="-", marker=next(marker), label=key)
    plt.legend(ncol=3)
    plt.show()
    f.savefig("results.png", bbox_inches='tight')
