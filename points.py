#!/usr/bin/env python3

from classify import *
from scipy import optimize

pi = np.pi

# poles for plotting similarity plot
def poles(n):
    th = np.arange(n)*2*pi/n
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

def find_bonds(u):
    n = len(u)
    B = bdist(u)
    B -= np.identity(n)

    #print(B)
    sk = set()
    bi = []
    bj = []
    k = []
    while len(sk) < n:
        ij = np.argmax(B)
        i = ij // n
        j = ij % n

        bi.append(i)
        bj.append(j)
        k.append(B[i,j])
        B[i,j] = 0
        B[j,i] = 0
        sk.add(i)
        sk.add(j)

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

def main(argv):
    assert len(argv) == 3, "Usage: %s <features.npy> <points.txt>"%argv[0]
    u = np.load(argv[1])
    bi, bj, k = find_bonds(u)

    x = poles(len(u))*2**0.5
    validate_deriv(x, bi, bj, k)

    def E(y):
        z = y.reshape(x.shape)
        return Eb(z,bi,bj,k) + Ei(z)
    def dE(y):
        z = y.reshape(x.shape)
        J = np.zeros(x.shape)
        dEb(J, z, bi, bj, k)
        dEi(J, z)
        return J.reshape(-1)

    ret = optimize.minimize(E, x, jac=dE, method='BFGS')
    x = ret.x.reshape(x.shape)
    print(x)
    with open(argv[2], "w") as f:
        for i,p in enumerate(x):
            f.write("%d %f %f\n"%(i+1,p[0],p[1]))

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

