#!/usr/bin/env python3

from classify import *

# redo categorization
def main(argv):
    assert len(argv) == 3, "Usage: %s <data.npy> <indices.npy>"%argv[0]
    ind = np.load(argv[2])
    B = BernoulliMixture(np.load("features.npy"), np.load("members.npy"))
    ind, x = load_features(argv[1], ind)
    z = B.classify(x)
    np.save("categories.npy", z)

if __name__=="__main__":
    import sys
    main(sys.argv)

