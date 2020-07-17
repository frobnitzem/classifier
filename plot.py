import numpy as np
import pylab as plt

#min_n = 50
min_n = 1

z = np.load("categories.npy").astype(int)
rmsd = np.load("../rmsdmat.npy")
if len(z)//10 == len(rmsd): # hack due to our particular trj.
    z = z[::10]
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

def plot_hist(c, n, ax, *args, **kws):
    dc = c[2]-c[1]
    ax.plot(c, n.astype(float)/(dc*n.sum()), *args, **kws)

def plt_pairs(rmsd, Nk, ax):
    nb, bins = np.histogram(rmsd.reshape(-1), 100)
    c = 0.5*(bins[1:]+bins[:-1])
    bins[0] += 0.001

    u = 0
    for i in range(K):
        if Nk[i] >= min_n:
            v = np.sum(Nk[:i+1])
            for j in range(i+1,K):
                if Nk[j] >= min_n:
                    n, _ = np.histogram(rmsd[u:u+Nk[i],v:v+Nk[j]].reshape(-1), bins)
                    plot_hist(c, n, ax, 'r-')
                v += Nk[j]
        u += Nk[i]
    u = 0
    for i in range(K):
        if Nk[i] >= min_n:
            n, _ = np.histogram(rmsd[u:u+Nk[i],u:u+Nk[i]].reshape(-1), bins)
            color = 'k-'
            #if i == 2:
            #    color = 'g-'
            plot_hist(c, n, ax, color)
        u += Nk[i]

    plot_hist(c, nb, ax, 'b-', linewidth=1.5)
    ax.set_xlabel("RMSD")
    ax.set_ylabel("PDF")

def plt_hist(x, Nk, ax):
    Nk = np.array(Nk)
    img = ax.imshow(x, cmap='bone_r')
    for p in Nk[:-1].cumsum():
        ax.axhline(p, linewidth=0.75, color='w')
        ax.axvline(p, linewidth=0.75, color='w')
    plt.colorbar(img)
    ax.set_xticks([])
    ax.set_yticks([])

fig, ax = plt.subplots(1, 2, figsize=(6.5,4))
x, Nk = permute(rmsd, z, K)

plt_pairs(x, Nk, ax[0])
ax[0].set_xlim(1.5, 5.5)
plt_hist(x, Nk, ax[1])
plt.savefig("rmsd.png")
plt.show()

