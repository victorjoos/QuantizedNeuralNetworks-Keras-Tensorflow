import matplotlib.pyplot as plt
import re

with open("results.out") as results:
    regex = re.compile(r'(.*), (.*), (.*), (.*)')
    
    for line in results:
        match = regex.match(line)
        type = match.group(1)
        acc, energy = match.group(3), match.group(4)

