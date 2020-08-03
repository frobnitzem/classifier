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
    np.save("probs.npy", P)
    z = P.argmax(1)
    np.save("categories.npy", z)

    ex = P.argmax(0) # exemplars for ea. category
    ptot = P.sum(0)  # sum of prob. for ea. category
    s = "# cluster, size, representative\n"
    for k,i in enumerate(ex):
        s += "%d %.0f %d\n"%(k+1, ptot[k], i+1)
    print(s)
    with open("rep.txt", "w") as f:
        f.write(s)

if __name__=="__main__":
    import sys
    main(sys.argv)

