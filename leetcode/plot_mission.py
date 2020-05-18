#coding:utf-8
import random
import matplotlib.pyplot as plt
import re


def plot_scatter(missions):
    plt.figure(figsize=(10, 6))
    for m in missions:
        x, y = missions[m][0], missions[m][1]
        if x + y >= 5:
            clr = 'r'
        elif x + y == 4:
            clr = 'b'
        else:
            clr = 'y'
        rd1, rd2 = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
        plt.scatter(x + rd1, y + rd2, marker='x', color=clr)
        plt.annotate(m, (x + rd1, y + rd2), fontsize=10)
    plt.ylabel('Importance')
    plt.xlabel('Urgency')
    plt.grid()
    plt.show()
    plt.savefig("missions.png")


if __name__ == "__main__":
    filename = "missions"
    with open(filename) as f:
        data = f.readlines()
    missions = {}
    for line in data:
        name = line.split()[2]
        quantities = list(map(int, re.findall('\d', line)[-2:]))
        missions.update({name: quantities})
    plot_scatter(missions)
