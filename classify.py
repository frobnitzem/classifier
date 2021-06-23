#!/usr/bin/env python3

from bernoulli import *
from pathlib import Path
from multiprocessing import Pool

#import cupy as cp
#rand = cp.random

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
        cat = B.sample_k(x)
        BBM = mk_BBMr(x, cat)
        B = BBM.sampleBernoulliMixture()
    print(B.logprior(True)) # inter-category distances
    print(BBM.Mj)

    return B, BBM

# run nchains replicas of gen_sample over x
# f : x, BBM -> a
# accum : *a -> a -> ()
def accum_sample(x, nchains, f, accum, *args, seed=None):
    seq = SeedSequence(seed)
    with Pool(nchains) as p:
        ans = p.map(run_sample, [(s, x, f, accum)+tuple(args) for s in seq.spawn(nchains)])
    for a in ans[1:]:
        accum(ans[0], a)
    return ans[0]

def run_sample(pack):
    s, x, f, accum, *args = pack
    ans = None
    for s in gen_sample(x, *args, seed=s):
        if ans is None:
            ans = f(x, s)
        else:
            accum(ans, f(x, s))
    return ans

def gen_sample(x, samples, skip=10, toss=500, seed=None):
    rand = default_rng(seed)
    BBM = BBMr(x.copy(), rand)
    acc = 0
    for S in range(1, samples+1):
        ok = BBM.morph()
        acc += ok
        BBM.recategorize()
        if S % (skip*10) == 0:
            print("Yielding sample %d"%(S/skip))
        if S > toss and S % skip == 0:
            yield BBM

    print("%d of %d moves accepted"%(acc,S))

from filelock import Timeout, FileLock

class Result:
    def __init__(self, x, BBM):
        self.count = [0]*BBM.K
        self.count[BBM.K-1] = 1
        self.best = [None]*BBM.K

        like = BBM.B.likelihood(BBM.x)
        self.best[BBM.K-1] = like

        # immediate outputs
        out = Path("sz%d"%BBM.K)
        out.mkdir(exist_ok=True)
        lock = FileLock(out / "info.txt.lock")
        with lock:
            f = out / "info.txt"
            if f.exists():
                with open(out / "info.txt") as f:
                    for line in f:
                        break
                lp = float(line.split()[-1])
            else:
                lp = -np.inf

            if lp < like:
                print("New best likelihood for K=%d: %e"%(BBM.K,like))
                print("Members: %s"%str(BBM.Nk))
                with open(out / "info.txt", "w") as f:
                    f.write("# log-likelihood = %e\n" % like
                            + '\n'.join("%2d %d"%(i+1,n) for i,n in enumerate(BBM.Nk))
                            + '\n')
                B = BBM.B
                np.save(out / "features.npy", B.p)
                np.save(out / "members.npy", B.c)

    def extend(self, other):
        d = len(other.count)-len(self.count)
        if d > 0:
            self.count += [0]*d
            self.best += [-np.inf]*d
        for i in range(len(other.count)):
            self.count[i] += other.count[i]
            if self.best[i] is None:
                self.best[i] = other.best[i]
            elif other.best[i] is not None \
                    and other.best[i] > self.best[i]:
                self.best[i] = other.best[i]

def accum(ans, a):
    ans.extend(a)

def main(argv):
    Usage = f"Usage {argv[0]} [--skip n] [--chains n] features.npy"

    skip = 1
    chains = 16
    ndx = None
    while len(argv) > 2:
        if argv[1] == "--skip":
            skip = int(argv[2])
            del argv[1:3]
        elif argv[1] == "--chains":
            chains = int(argv[2])
            del argv[1:3]
        elif argv[1] == "--index":
            ndx = np.load(argv[2])
            del argv[1:3]
        else:
            break
    assert skip > 0, Usage
    assert chains > 0, Usage
    assert len(argv) == 2, Usage

    best = [] # best likelihood at each n

    if skip != 1:
        sl = slice(None, None, skip)
        ind, x = load_features(argv[1], ndx, sl=sl)
    else:
        ind, x = load_features(argv[1], ndx)
    print("Loaded %d x %d feature matrix"%x.shape)
    np.save("indices.npy", ind)

    #z = cluster(x, 10)
    #print("Created %d clusters for %d data points."%(z.max()+1, len(x)))
    #BBM = mk_BBMr(x, z)

    R = accum_sample(x, chains, Result, accum, 1500)
    if False:
      R = None
      for BBM in gen_sample(x, 10*1000, skip=10, toss=500):
        if R is None:
            R = Result(x, BBM)
        else:
            R2 = Result(x, BBM)
            R.extend(R)

    print("Counts: %s"%str(R.count))
    print("Best log-likelihood: %s"%str(R.best))

if __name__=="__main__":
    import sys
    main(sys.argv)

