#!/usr/bin/env python3

from ucgrad import *

def main(argv):
    assert len(argv) == 4, "Usage: %s <prot.pdb> <beta.npy> <out.pdb>"%argv[0]

    pdb, x = read_pdb(argv[1])
    ca = pdb.atname("CA")
    ca = list(ca)
    ca.sort()

    beta = np.load(argv[2])
    beta = np.abs(beta) * (9.99/beta.max())
    pdb.ob[ca,1] = beta
    pdb.write(argv[3])

if __name__=="__main__":
    import sys
    main(sys.argv)
