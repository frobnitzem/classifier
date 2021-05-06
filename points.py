#!/usr/bin/env python3

from classify import *
from scipy import optimize
rand = np.random

pi = np.pi

# poles for plotting similarity plot
def poles(perm):
    n = len(perm)
    th = np.array(perm)*2.0*pi/n
    poles = np.zeros((n,2))
    poles[:,0] = np.cos(th)
    poles[:,1] = np.sin(th)
    return poles

def Eb(x, bi, bj, k = 2.0, r0=1.0):
    xi = x[bi]
    xj = x[bj]
    rij = xi - xj
    r2 = (rij*rij).sum(-1)
    r = np.sqrt(r2)
    E = k*(r - r0)**2
    return 0.5*E.sum()

# U = k/2 (|r| - r0)^2
# dU/dr = k (|r| - r0) d|r|/dx = -k(|r|-r0) r/|r|
def dEb(dU, x, bi, bj, k = 2.0, r0=1.0):
    xi = x[bi]
    xj = x[bj]
    rij = xi - xj
    r2 = (rij*rij).sum(-1)
    r = np.sqrt(r2)
    #E = k*(r - r0)**2
    #for i,j,ki,rij,Ei in zip(bi,bj,k,r,E):
    #    print(i,j,ki,rij,Ei)
    dE = (k*(1.0-r0/r))[:,newaxis]*rij # dU/dxi
    for i,j,dEi in zip(bi,bj,dE):
        dU[i] += dEi
        dU[j] -= dEi
    #dU[bi] += dE
    #dU[bj] -= dE

def Ei(x):
    n = len(x)
    dr = x[:,newaxis] - x[newaxis,:]
    r2 = (dr*dr).sum(-1)
    E = 0.0
    for i in range(n-1):
        E += np.log(r2[i,i+1:]).sum()
    return -0.5*E

# U = -ln |r|
# dU = -r/|r|^2
def dEi(dU, x):
    n = len(x)
    dr = x[:,newaxis] - x[newaxis,:]
    r2 = (dr*dr).sum(-1)
    for i in range(n-1):
        dE = -dr[i,i+1:] / r2[i,i+1:,newaxis]
        dU[i] += dE.sum(0)
        dU[i+1:] -= dE
        #for j in range(i+1, n):
        #    dE = -dr[i,j]/r2[i,j]
        #    dU[i] += dE
        #    dU[j] -= dE

# 1 for distributions that are alike
def bdist(u):
    return np.prod( np.sqrt(u[:,newaxis]*u[newaxis,:]) + np.sqrt((1-u)[:,newaxis]*(1-u)[newaxis,:]), -1 )

def find_subset(sk, i):
    for k,s in enumerate(sk):
        if i in s:
            return k
    return None

# Modify sets to add an edge, i-j
# this is an inefficient union-find.
def add_sets(sk, i, j):
    si = find_subset(sk, i)
    sj = find_subset(sk, j)
    if si is None: # if either si,sj is known, it's i
        i,j = j,i
        si,sj = sj,si
    if si == sj: # already in the same set
        if si is None:
            sk.append(set([i,j]))
        return len(sk[0])
    # si is known
    if sj is None:
        sk[si].add( j )
        return len(sk[0])
    # two separate sets
    if si > sj: # let si be the smaller one
        i,j = j,i
        si,sj = sj,si
    sk[si] |= sk.pop(sj)
    return len(sk[0])

def find_bonds(u):
    n = len(u)
    B = bdist(u)
    B -= np.identity(n)

    #print(B)
    sk = []
    bi = []
    bj = []
    k  = []
    order = [] # order of found points
    while True:
        ij = np.argmax(B)
        i = ij // n
        j = ij % n

        bi.append(i)
        bj.append(j)
        k.append(B[i,j])
        B[i,j] = 0
        B[j,i] = 0
        if add_sets(sk, i, j) == n:
            break

    bi = np.array(bi, int)
    bj = np.array(bj, int)
    k = np.array(k)
    return bi, bj, k

def err(x, y):
    #print(x)
    #print(y)
    e = (np.abs(x-y)).max() / np.abs(x).max()
    #print("Relative err = %e"%e)
    assert e < 1e-4

def validate_deriv(x, bi, bj, k):
    from ucgrad import Ndiff
    dE1 = Ndiff(x, lambda y: Eb(y, bi, bj, k))
    dE2 = np.zeros(x.shape)
    dEb(dE2, x, bi, bj, k)
    err(dE1, dE2)

    dE1 = Ndiff(x, Ei)
    dE2 = np.zeros(x.shape)
    dEi(dE2, x)
    err(dE1, dE2)

def get_cfg(perm, bi, bj, k):
    x = poles(perm)

    def E(y):
        z = y.reshape(x.shape)
        return Eb(z,bi,bj,k) + Ei(z)
    def dE(y):
        z = y.reshape(x.shape)
        J = np.zeros(x.shape)
        dEb(J, z, bi, bj, k)
        dEi(J, z)
        return J.reshape(-1)

    #validate_deriv(x, bi, bj, k)
    ret = optimize.minimize(E, x, jac=dE, method='BFGS')
    return ret.x.reshape(x.shape), ret.fun

def write_x(name, x):
    with open(name, "w") as f:
        for i,p in enumerate(x):
            f.write("%d %f %f\n"%(i+1,p[0],p[1]))

def main(argv):
    circ = False
    if len(argv) > 3 and argv[1] == "--circ":
        circ = True
        del argv[1]
    assert len(argv) == 3, "Usage: %s <features.npy> <points.txt>"%argv[0]
    u = np.load(argv[1])

    if circ:
        x = poles(np.random.permutation(len(u)))
        write_x(argv[2], x)
        return 0

    bi, bj, k = find_bonds(u)

    xmin, Emin = get_cfg(range(len(u)), bi, bj, k)
    for i in range(10):
        x, E = get_cfg(np.random.permutation(len(u)), bi, bj, k)
        print(E)
        if E < Emin:
            xmin, Emin = x, E
    print(xmin)

    write_x(argv[2], x)
    try:
        import pylab as plt
        plt.plot(x[:,0], x[:,1], 'ko', fillstyle='none', markersize=18)
        for i,p in enumerate(x):
            plt.text(p[0], p[1], str(i+1), color="black", fontsize=12,
                     horizontalalignment='center', verticalalignment='center')
        plt.show()
    except ImportError:
        pass


if __name__=="__main__":
    import sys
    main(sys.argv)

