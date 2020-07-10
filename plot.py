import numpy as np
import pylab as plt

z = np.load("categories.npy").astype(int)
rmsd = np.load("../rmsdmat.npy")
if len(z) == len(rmsd)+1: # hack due to our particular trj.
    z = z[1:]

S = len(z)
assert rmsd.shape == (S,S)

K = z.max()+1

def permute(rmsd, z, K):
    S = len(z)
    assert rmsd.shape == (S,S)
    Nk = [np.sum(z == k) for k in range(K)]
    assert np.sum(Nk) == S
    print("%d categories: %s"%(S,str(Nk)))

    x = np.zeros(rmsd.shape)
    u = 0
    for i in range(K):
        if Nk[i] == 0: continue
        x[u:u+Nk[i],u:u+Nk[i]] = rmsd[z==i][:,z==i]
        v = np.sum(Nk[:i+1])
        for j in range(i+1,K):
            if Nk[j] == 0: continue
            x[u:u+Nk[i],v:v+Nk[j]] = rmsd[z==i][:,z==j]
            x[v:v+Nk[j],u:u+Nk[i]] = rmsd[z==j][:,z==i]
            v += Nk[j]
        u += Nk[i]

    return x, Nk

def plt_pairs(rmsd, Nk):
    n, bins = np.histogram(rmsd.reshape(-1), 100)
    bins[0] += 0.001
    c = 0.5*(bins[1:]+bins[:-1])
    n = n.astype(float)/n.sum()
    plt.plot(c, n, 'b-')

    u = 0
    for i in range(K):
        if Nk[i] == 0: continue
        n, _ = np.histogram(rmsd[u:u+Nk[i],u:u+Nk[i]].reshape(-1), bins)
        n = n.astype(float)/n.sum()
        plt.plot(c, n, 'k-')
        v = 0
        for j in range(i+1,K):
            n, _ = np.histogram(rmsd[u:u+Nk[i],v:v+Nk[j]].reshape(-1), bins)
            n = n.astype(float)/n.sum()
            plt.plot(c, n, 'r-')
            v += Nk[j]
        u += Nk[i]
    plt.show()

x, Nk = permute(rmsd, z, K)
plt.imshow(x)
plt.colorbar()
plt.show()

plt_pairs(x, Nk)
