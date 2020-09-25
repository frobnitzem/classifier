#!/usr/bin/env python3

import numpy as np

def scale_eps(y, eps):
    return (1-eps)*y + eps

def main(argv):
    assert len(argv) == 3, "Usage: %s <feat.npy> <diffs.npy>"%argv[0]
    f = np.load(argv[1])

    N = len(f)
    betas = np.zeros((N*(N-1)//2, f.shape[1]), np.float)
    k = 0
    for i in range(N-1):
      for j in range(i+1, N):
        beta = betas[k]
        k += 1

        a = f[i]
        b = f[j]
        dmu = np.sqrt(a*b) + np.sqrt((1-a)*(1-b))
        eps = np.exp(-10.0) # max difference = +10
        beta[:] = np.abs(np.log(scale_eps(dmu, eps))) # larger = more different
        print(i+1,j+1,beta.min(),beta.max())

    np.save(argv[2], betas)

if __name__=="__main__":
    import sys
    main(sys.argv)
