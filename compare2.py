#import numpy as np
from bernoulli import *

class FeaturePair:
    def __init__(self, ind1, ind2, x1, x2):
        print(x1.shape, x2.shape)
        s1 = set(ind1)
        s2 = set(ind2)

        iset = list(s1 | s2)
        iset.sort()
        idx = dict((j,i) for i,j in enumerate(iset))

        # permutation to combined index space
        self.P = np.array(
                   [ [idx[j] for j in ind1]
                   , [idx[j] for j in ind2]
                   ])

        # excluded categories
        miss1 = list(s2 - s1)
        miss1.sort()
        miss2 = list(s1 - s2)
        miss2.sort()
        self.xP = np.array(
                    [ [idx[j] for j in miss1]
                    , [idx[j] for j in miss2]
                    ])
        self.x  = [ x1[miss1], x2[miss2] ]

        self.iset = np.array(iset)
        self.M = len(iset)

        # shared index set
        self.shared = np.array([idx[j] for j in (s1&s2)])
        self.shared.sort()

    def expand(self, B, j):
        K = len(B)
        P = self.P[j]
        xP = self.xP[j]

        B2 = np.zeros((K, self.M))
        B2[:,P]  = B
        B2[:,xP] = self.x[j]

        return B2

def dist(u, v):
    B = np.prod( np.sqrt(u[:,newaxis,:]*v[newaxis,:,:])
               + np.sqrt((1-u)[:,newaxis,:]*(1-v)[newaxis,:,:]),
                2)
    return B

if __name__=="__main__":
    import sys
    argv = sys.argv
    assert len(argv) == 7, "Usage: %s <ind1.npy> <ind2.npy> <x1.npy> <x2.npy> <feat1.npy> <feat2.npy>"%argv[0]
    P = FeaturePair(np.load(argv[1]), np.load(argv[2]),
                    np.unpackbits(np.load(argv[3])[0]),
                    np.unpackbits(np.load(argv[4])[0])
                   )
    f1 = P.expand(np.load(argv[5]), 0)
    f2 = P.expand(np.load(argv[6]), 1)
    print(f1)
    print(f2)
    print("1:1")
    print(dist(f1, f1))
    print("2:2")
    print(dist(f2, f2))
    print("1:2")
    print(dist(f1, f2))

