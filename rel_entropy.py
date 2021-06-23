#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def lclip(y, eps):
    return y + (eps-y)*(y < eps)

# exp( D(y || x) )
def rel_ent(x, y, eps=1e-17):
    return np.exp( np.dot(     x, np.log(lclip(    y, eps)/lclip(   x, eps))) \
                 + np.dot( 1.0-x, np.log(lclip(1.0-y, eps)/lclip(1.0-x, eps))))

def rel_B(x,y):
    return np.prod( np.sqrt(x*y) + np.sqrt((1-x)*(1-y)) )

# permutation from A (row index) onto B (col index)
def re_order(M):
    A = set()
    B = set()
    ms = []
    for i,row in enumerate(M):
        ms.extend( [(x,i,j) for j,x in enumerate(row)] )
    ms.sort(reverse=True)

    #perm = [ row.argmax() for row in M ] # simple permutation (replacement)
    perm = [ -1 ]*len(M)

    # remove matched values
    while len(A) < M.shape[0] and len(B) < M.shape[1]:
        _,i,j = ms.pop(0)
        if i in A or j in B:
            continue
        A.add(i)
        B.add(j)
        perm[i] = j

    return perm

def main(argv):
    assert len(argv) == 4, "Usage: %s <feat1.npy> <feat2.npy> <out.pdf>"%argv[0]
    A = np.load(argv[1])
    B = np.load(argv[2])

    N = len(A)
    M = len(B)

    S = np.zeros((N,M), float)
    D = np.zeros((N,M), float)
    for i in range(N):
        for j in range(M):
            S[i,j] = rel_ent(A[i], B[j])
            D[i,j] = rel_B(A[i], B[j])

    xt = np.arange(1, N+1, N//5)
    yt = np.arange(1, M+1, M//5)
    yl = np.arange(1, M+1, M//5)
    if True:
        # permutation from B (y-axis / col) onto A (x-axis / row)
        perm = re_order( S.transpose() )
        print(perm)
        Sp = S.copy()
        Dp = D.copy()
        yt = np.arange(1, M+1)
        yl = np.arange(1, M+1)
        S[:] = 0.0
        D[:] = 0.0
        extra = min(N,M)
        for j,i in enumerate(perm):
            if i == -1:
                i = extra
                extra += 1
            S[:,i] = Sp[:,j]
            D[:,i] = Dp[:,j]
            yl[i] = j

    fig, ax = plt.subplots(1,2)

    for a, Z in zip(ax, [S,D]):
        img = a.imshow(Z.transpose(), origin='upper', cmap='Reds',
                        vmin=0, vmax=1,
                        interpolation='nearest', extent=[0.5,N+0.5,0.5,M+0.5])
        a.set_xticks(xt)
        a.set_yticks(yt)
        a.set_yticklabels(yl)

    #fig.colorbar(img)

    plt.savefig(argv[3])
    plt.close()

if __name__=="__main__":
    import sys
    main(sys.argv)
