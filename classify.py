#!/usr/bin/env python3

from bernoulli import *
from pathlib import Path

#import cupy as cp
#rand = cp.random

class Category:
    def __init__(self, Mj=None, N=None, M=None):
        if Mj is None:
            assert M is not None
            assert N is None
            self.Mj = np.zeros(M, np.int)
            self.N = 0
        else:
            assert N is not None
            self.Mj = np.array(Mj, np.int)
            self.N = N

    def append(self, x):
        self.Mj += x
        self.N += 1
    def concat(self, x):
        self.Mj += x.sum(0)
        self.N += len(x)

    def __add__(L, R):
        return Category(L.Mj+R.Mj, L.N+R.N)

    def ldist(L, R, N=None, K=None): # log(P[different cat] / P[same cat])
        if N is None:
            N = L.N+R.N
        if K is None:
            K = 1
        return calc_Qxy(L.Mj, L.N, R.Mj, R.N, N, K)

def hamming(x):
    N = x.shape[0]

    chunk = 128
    H = np.zeros((N,N), np.int)
    for i in range(0, N, chunk):
        j = min(i+chunk, N)
        H[i:j] = np.sum(x[i:j,newaxis,:]^x[newaxis,:,:], 2)

    return H

# add samples to nearest clusters in serial order
# -473 to -343.7454308856099
def cluster(x, max_clusters, lrat = -400.0):
    N = x.shape[0]
    #M = x.shape[1]
    lst = np.arange(N)
    for i in range(N-1): # random addition order
        j = int(uniform()*(len(x)-i))
        lst[i], lst[j] = lst[j], lst[i]

    z = np.zeros(N, np.int)
    clusters = [Category(x[0], 1)]
    K = 1
    S = 1
    for i in lst[1:]:
        S += 1
        u = Category(x[i], 1)
        dst = np.array([c.ldist(u, S, K) for c in clusters])
        v = dst.argmin() # closest category
        if K < max_clusters and dst[v] > lrat:
            z[i] = len(clusters)
            clusters.append(u)
            K += 1
        else:
            z[i] = v
            clusters[v].append(x[i])
    return z

class BBMs:
    def __init__(self, x, Nk, min_lk=0.9):
        S = x.shape[0]
        M = x.shape[1]
        Nk = np.array( [ 3.0 ] )
        Mj = np.array( [ 1.0 + x[0] ] )
        #B = BernoulliMixture(Mj/Nk[:,newaxis], Nk/sum(Nk), M+eta)
        for i in range(1, len(x)):
            p = Mj/Nk[:,newaxis]
            P = np.prod( p*x[i,newaxis,:] + (1-self.p)*(1-x[i,newaxis,:]) , 2 )**(1.0/M)
            cat = P.argmax()
            if P[cat] < min_lk: # create a new category and continue
                Nk
                continue
            Nk[cat] += 1
            Mj[cat] += x

def plot(x, Nk, P):
    K = P.shape[1]
    if K == 1:
        return
    poles = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5*3.0**0.5]])
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if K > 3:
        xy = np.dot(P[:,:3], poles)
    elif K == 2:
        xy = np.dot(P, poles[:2])
    else:
        xy = np.dot(P, poles)
    xy += 0.07*(rand.random(xy.shape)-0.5) # jitter
    n = 0
    for m, c in zip(Nk, ['b', 'r', 'k']):
        ax.scatter(xy[n:n+m,0], xy[n:n+m,1], c=c, alpha=0.03)
        n += m
    plt.show()

def bootstrap(x, Nguess):
    # Jump-start an unsupervised model based on 100 trials.
    S = x.shape[0]
    M = x.shape[1]
    # default model = single category
    B1  = BernoulliMixture( (np.sum(x,0)/float(S)).reshape((1,M)), np.ones(1))
    lk1 = B1.likelihood(x)
    for i in range(Nguess):
        lm = cat_penalty / 10.0
        K = int( np.ceil(-np.log(uniform()+1e-8)/lm) ) + 1
        # random categories
        B = BernoulliMixture(rand.random((K,M)), np.ones(K)/float(K), M+eta)
        cat = B.classify(x)
        BBM = mk_BBMr(x, cat)
        B = BBM.sampleBernoulliMixture() # likely sample from B

        lk = B.likelihood(x)
        if lk > lk1:
            lk1 = lk
            B1 = B
    return B1

def sample(B, x, Niter):
    # sample burn-in starting from best model
    for i in range(Niter):
        cat = B.classify(x)
        BBM = mk_BBMr(x, cat)
        B = BBM.sampleBernoulliMixture()
    print(B.logprior(True)) # inter-category distances
    print(BBM.Mj)

    return B, BBM

def main(argv):
    best = [] # best likelihood at each n

    ind, x = load_features(argv[1])
    np.save("indices.npy", ind)

    #z = cluster(x, 10)
    #print("Created %d clusters for %d data points."%(z.max()+1, len(x)))

    #BBM = mk_BBMr(x, z)
    BBM = BBMr(x)
    acc = 0
    for i in range(10*1000):
        #BBM.recategorize()
        B = BBM.sampleBernoulliMixture()
        prob = B.likelihood(BBM.x)

        z = B.classify(BBM.x)
        y, Nk = reshuffle(BBM.x, z)
        BBM.x[:] = y
        BBM.recompute(Nk)
        if len(best) < BBM.K or prob > best[BBM.K-1]:
            if len(best) < BBM.K:
                best.append(0)
            best[BBM.K-1] = prob
            print("New best likelihood: %e"%prob)
            print("Members: %s"%str(Nk))
            out = Path("sz%d"%BBM.K)
            out.mkdir(exist_ok=True)
            with open(out / "info.txt", "w") as f:
                f.write("# log-likelihood = %e\n"%prob
                        + '\n'.join("%2d %d"%(i+1,n) for i,n in enumerate(Nk))
                        + '\n')
            np.save(out / "features.npy", B.p)
            np.save(out / "members.npy", B.c)

        acc += BBM.morph()

        if (i+1)%1000 == 0:
            print("Sample %d."%(i+1))

    print("%d of %d moves accepted"%(acc,i+1))
    print(BBM.Nk)
    print(BBM.Mj)
    del BBM

if __name__=="__main__":
    import sys
    main(sys.argv)

