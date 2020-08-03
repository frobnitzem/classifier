#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

pi = np.pi

# poles for plotting similarity plot
def poles(n):
    if n == 2: # degenerate situations
        return np.identity(2)

    th = np.arange(n)*2*pi/n
    poles = np.zeros((n,2))
    poles[:,0] = np.cos(th)
    poles[:,1] = np.sin(th)
    return poles

def plot(P, out):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)

    xy = np.dot(P, poles(P.shape[1]))

    ax.scatter(xy[:,0], xy[:,1],
               c='b', alpha=min(1.0, 150/len(P)))
    plt.savefig(out)
    plt.show()

def main(argv):
    assert len(argv) == 3, "Usage: %s <probs.npy> <probs.pdf>"%argv[0]

    plot(np.load(argv[1]), argv[2])

if __name__ == "__main__":
    import sys
    main(sys.argv)

