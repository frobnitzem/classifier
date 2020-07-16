#!/usr/bin/env python3

from ucgrad import *

def main(argv):
    assert len(argv) == 5, "Usage: %s <id list> <pdb> <dcd> <cluster_out.pdb>"%argv[0]
    num = []
    with open(argv[1]) as f:
        for line in f:
            line = line.split('#',1)[0].strip()
            tok = line.split()
            if len(tok) != 3:
                continue
            num.append(int(tok[2])-1)

    pdb, x = read_pdb(argv[2])
    dcd = read_dcd(argv[3])
    f = open(argv[4], "w")

    for k,i in enumerate(num):
        dcd.seek(i)
        for cell, x in dcd:
            break
        pdb.L[:] = cell
        pdb.x[:] = x
        pdb.write(f, model=k+1)
    f.close()

if __name__=="__main__":
    import sys
    main(sys.argv)
