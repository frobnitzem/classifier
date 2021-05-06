#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def plot(P, poles, out):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    xy = np.dot(P, poles)

    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    while P.shape[1] > len(names):
        names = names+names
    #assert P.shape[1] <= len(names), "Not enough colors to color ea. point!"
    names = names[:P.shape[1]]

    z = P.argmax(1)
    for k, c in enumerate(names):
        m = (z == k)
        ax.scatter(xy[z == k,0], xy[z == k,1],
                   c=colors[c], alpha=min(1.0, 150.0/min(2e4,m.sum())))

    #plt.plot(x[:,0], x[:,1], 'ko', fillstyle='none', markersize=18)
    for i,p in enumerate(poles):
        plt.text(p[0], p[1], str(i+1), color="black", fontsize=12,
                 horizontalalignment='center', verticalalignment='center')

    plt.savefig(out)
    #plt.show()

def main(argv):
    assert len(argv) == 4, "Usage: %s <probs.npy> <points.txt> <probs.pdf>"%argv[0]

    P = np.load(argv[1])
    poles = np.fromfile(argv[2], sep=" ").reshape((P.shape[1],3))[:,1:]
    plot(P, poles, argv[3])

if __name__ == "__main__":
    import sys
    main(sys.argv)

