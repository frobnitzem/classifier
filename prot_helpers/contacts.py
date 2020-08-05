#!/usr/bin/env python3

from ucgrad import *
import numpy as np
import imp

# Contacts: computes features from simulation trajectory frames.
#
# dists.py contains a function, selections,
# which takes pdb as an input and returns N (the number of features)
# and feat (a generator which computes the features, when provided coordinates -- (L,x))
#
#   selections : pdb -> (int, feat_type)
#   feat : array((3,3), float) -> array(pdb.x.shape, float) -> generator( array(M < N, bool) )
#
#   each time feat runs, it must provide N total features [sum(M) == N]
#
def main(argv):
    assert len(argv) == 5, "Usage: %s <prot.pdb> <dists.py> <prot.dcd> <contacts.npy>"%argv[0]

    pdb, x = read_pdb(argv[1])
    dists = imp.load_source('dists', argv[2])
    N, feat = dists.selections(pdb) # count of features and selector object
    dcd = read_dcd(argv[3])

    u = np.zeros(N, np.bool)
    ct = np.zeros((dcd.frames,(N+7)//8), np.uint8) # gets 0-padded at right side
    for i, (L, x) in enumerate(dcd):
        k = 0
        for f in feat(L, x):
            M = len(f)
            u[k:k+M] = f
            k += M
        assert k == N, "Invalid number of features."
        ct[i] = np.packbits(u)
    np.save(argv[4], ct)

if __name__=="__main__":
    import sys
    main(sys.argv)
