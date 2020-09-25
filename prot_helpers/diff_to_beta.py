#!/usr/bin/env python3

from ucgrad import *
import numpy as np

# diff_to_beta: Projects feature values down to beta
def main(argv):
    assert len(argv) == 5, "Usage: %s <clusters.pdb> <lst_ij.npy> <indices.npy> <diffs.npy>"%argv[0]

    pdb, x = read_pdb(argv[1])
    lst_ij = np.load(argv[2]) # count of features and selector object
    assert len(lst_ij.shape) == 2 and lst_ij.shape[1] == 2
    ind = np.load(argv[3])
    mus = np.load(argv[4])
    assert len(mus.shape) == 2 and mus.shape[0] == len(x)*(len(x)-1)//2
    assert mus.shape[1] == len(ind), "Incommensurate indices.npy and diff.npy"

    ck = 0
    for ci in range(len(x)-1):
      for cj in range(ci+1, len(x)):
        mu = mus[ck]
        ck += 1
        beta = np.zeros(len(pdb.x), np.float)
        for k,z in zip(ind, mu):
          i, j = lst_ij[k]
          beta[i] += z
          beta[j] += z

        beta = np.abs(beta) * (9.99/beta.max())
        pdb.ob[:,1] = beta
        pdb.x[:] = x[ci]
        pdb.write("diff_%d_%d.pdb"%(ci+1,cj+1), 'w', ci+1)
        pdb.x[:] = x[cj]
        pdb.write("diff_%d_%d.pdb"%(ci+1,cj+1), 'a', cj+1)

if __name__=="__main__":
    import sys
    main(sys.argv)
