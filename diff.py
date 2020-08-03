#!/usr/bin/env python3

#import numpy as np
from bernoulli import *

# Iterative Binary Search Function
# returns 0 <= i < N such that f(i) <= x < f(i+1)
# f(i) must be strictly increasing
def binary_search(f, x, N):
    low = 0
    high = N-1
    while low < high-1:
        mid = (high + low) // 2
        fx = f(mid)
        if fx < x:
            low = mid
        elif fx > x:
            high = mid
        else:
            return mid
    return low

def idx(k, N):
    s = lambda n: n*N - n*(n+1)//2
    i = binary_search(s, k, N)
    j = i+1+(k-s(i))
    return i, j

# Project the pair-contact probability vector, mu,
# onto the index space, i
# where iset[n] corresponds to mu[n]
# and iset[n] = s(i)+j-i-1, a multi-index for pairs 0 <= i < j < N
# and    s(i) = i*(N-(i+1)/2)
def proj(iset, mu, N):
    beta = np.zeros(N, np.float)
    for k,z in zip(iset, mu):
        i, j = idx(k, N)
        beta[i] += z
        beta[j] += z
    return beta

def scale_eps(y, eps):
    return (1-eps)*y + eps

def main(argv):
    assert len(argv) == 4, "Usage: %s <x.npy> <ind.npy> <feat.npy>"%argv[0]
    x = np.unpackbits(np.load(argv[1])[0])
    M = len(x)
    N = binary_search(lambda n: n*(n-1)//2, M, M)
    print(N, M, N*(N-1)//2)

    ind = np.load(argv[2])
    f = np.load(argv[3])

    for i in range(len(f)-1):
      for j in range(i+1, len(f)):
        a = f[i]
        b = f[j]
        dmu = np.sqrt(a*b) + np.sqrt((1-a)*(1-b))
        eps = np.exp(-10.0)
        beta = -proj(ind, np.log(scale_eps(dmu, eps)), N)

        #beta = -np.log( scale_eps(np.exp(beta), eps) )
        print(i,j)
        print(beta.max())
        np.save("beta_%d_%d.npy"%(i,j), beta)

if __name__=="__main__":
    import sys
    main(sys.argv)
