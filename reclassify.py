#!/usr/bin/env python3

from classify import *

# redo categorization
def main(argv):
    assert len(argv) == 2, "Usage: %s <data.npy>"%argv[0]
    B = BernoulliMixture(np.load("features.npy"), np.load("members.npy"))
    x = load_features(argv[1])
    z = B.classify(x)
    np.save("categories.npy", z)

if __name__=="__main__":
    import sys
    main(sys.argv)

