#!/usr/bin/env python3

from classify import *

# redo categorization
def main(argv):
    assert len(argv) == 3, "Usage: %s <data.npy> <indices.npy>"%argv[0]
    ind = np.load(argv[2])
    ind, x = load_features(argv[1], ind)
    M = x.shape[1]
    B = BernoulliMixture(np.load("features.npy"), np.load("members.npy"), M+1, None)
    P = B.categorize(x)
    z = P.argmax(1)
    np.save("categories.npy", z)

    ex = P.argmax(0) # exemplars for ea. category
    print("# cluster, size, representative")
    for k,i in enumerate(ex):
        print("%d %d %d"%(k+1, np.sum(z == k), i+1))

if __name__=="__main__":
    import sys
    main(sys.argv)

