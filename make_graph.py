import matplotlib.pyplot as plt
import re
from collections import defaultdict
import itertools


with open("results.out") as results:
    regex = re.compile(r'(.*), (.*), (.*), (.*)')
    accs = defaultdict(list)
    Es = defaultdict(list)
    for line in results:
        match = regex.match(line)
        type = match.group(1)
        acc, energy = float(match.group(3)), float(match.group(4))
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
