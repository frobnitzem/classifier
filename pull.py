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
    f = open(argv[4], "w")
    fname = argv[3]
    if fname[-4:] == ".dcd":
        g = from_dcd(fname, num)
    else:
        g = from_xdr(fname, num)

    k = 1
    for cell, x in g:
        pdb.L = cell
        pdb.x[:] = x
        pdb.write(f, model=k)
        k += 1
    f.close()

def from_dcd(fname, num):
    dcd = read_dcd(fname)
    for k,i in enumerate(num):
        dcd.seek(i)
        for cell, x in dcd:
            break
        yield cell, x

def from_xdr(fname, num):
    from xdrfile.xdrfile import xdrfile

    trr = xdrfile(fname)
    nn = [(n,k) for k,n in enumerate(num)]
    nn.sort() # have to go in-order

    # gather all outputs
    out = [0]*len(nn)
    k = 0
    for i,f in enumerate(trr):
        if i == nn[k][0]:
            out[nn[k][1]] = (f.box.copy(), f.x.copy())
            k += 1
            if k == len(out):
                break
    else:
        raise RuntimeError("Not all confs found!")

    # iterate over outputs
    for o in out:
        yield o

if __name__=="__main__":
    import sys
    main(sys.argv)

