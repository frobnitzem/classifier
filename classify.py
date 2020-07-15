#!/usr/bin/env python3

from prob import *
# imports eta

from pathlib import Path

#import cupy as cp
#rand = cp.random

# Mixture of independent multivariate Bernoulli distributions.
class BernoulliMixture:
    def __init__(self, p, c, alpha=None):
        assert len(p.shape) == 2 and len(c.shape) == 1
        self.K = len(c)
        assert len(p) == self.K
        self.M = p.shape[1]
        if alpha is None:
            self.alpha = self.M+eta
        else:
            self.alpha = alpha

        self.p = p
        self.c = c

    def logprior(self, verb=False):
        # Calculate a prior probability over categories from a
        # matrix of Bhattacharyya distances
        p = self.p
        B = np.prod( np.sqrt(p[:,newaxis,:]*p[newaxis,:,:])
                    + np.sqrt((1-p)[:,newaxis,:]*(1-p)[newaxis,:,:]),
                    2)
        if verb:
            print(B)
        B -= np.identity(self.K)
        # All K choose 2 categories differ:
        return 0.5*np.log(1-B).sum()

    def d_logprior(self):
        p = self.p
        dlp = np.zeros((self.K,self.K,self.M))
        pd  = np.sqrt(p[:,newaxis,:]*p[newaxis,:,:]) \
                    + np.sqrt((1-p)[:,newaxis,:]*(1-p)[newaxis,:,:])
        for j in range(self.M):
            u = pd[:,:,j].copy()
            pd[:,:,j] = np.sqrt(p[newaxis,:,j]/(1e-10*(p[:,j]<1e-10)+p[:,j])[:,newaxis]) \
                        - np.sqrt((1-p[:,j])[newaxis,:]/(1+1e-10*(p[:,j]+1e-10>1)-p[:,j])[:,newaxis])
            dlp[:,:,j] = np.prod(pd, 2)
            pd[:,:,j] = u

        B = np.prod(pd, 2)
        B -= np.identity(self.K)
        dlp /= 1.0-B[:,:,newaxis]
        # All K choose 2 categories differ:
        return -0.5*np.sum(dlp, 1)

    # Log-likelihood of a sample, x, given this model
    def likelihood(self, x):
        P = np.dot(np.prod( self.p*x[:,newaxis,:] + (1-self.p)*(1-x[:,newaxis,:]) , 2 ), self.c)
        lP = np.sum(np.log(P)) + self.logprior() + cat_prior(len(x), self.alpha, self.M, self.K) \
               + (self.alpha-1)*np.log(self.c).sum()
        return lP

    # derivative of log-likelihood of a sample, x, given this model
    def d_like(self, x):
        dlP = self.d_logprior()
        P1 = self.p*x[:,newaxis,:] + (1-self.p)*(1-x[:,newaxis,:]) # N,K,M
        P2 = np.prod(P1, 2) # N,K
        D  = 1./np.dot(P2, self.c) # N
        dlC = np.dot(D, P2) + (self.alpha-1)/self.c
        for j in range(self.M):
            u = P1[:,:,j].copy()
            P1[:,:,j] = 2*x[:,newaxis,j] - 1
            dlP[:,j] += self.c * np.dot(D, np.prod(P1, 2))
            P1[:,:,j] = u
        return dlP, dlC

    def sample(self, N):
        # N : int^K
        # generate N[k] independent samples from each category, k
        assert len(N) == self.K
        S = np.sum(N) # total number of samples
        x = np.zeros((S,self.M), int)
        k = 0
        for pk, Nk in zip(self.p, N):
            x[k:k+Nk] = Bernoulli(pk, Nk)
            k += Nk
        return x

    def categorize(self, x):
        # compute P(k | x, p, c) : S by K
        # note - this only works for x = 0 or 1
        # x is S by M
        # p is K by M
        P = np.prod( self.p*x[:,newaxis,:] + (1-self.p)*(1-x[:,newaxis,:]) , 2 ) * self.c
        P /= np.sum(P, 1)[:,newaxis]
        return P

    # calculate maximum likelihood categorization
    def classify(self, x):
        P = self.categorize(x)
        return P.argmax(1) # Max-likelihood category

    def maximize(self):
        # TODO: implement mapping p1 = exp(x1)/(1+exp(x1)+exp(x2)), etc.
        pass

    # pull a category sample
    def sample_k(self, x):
        P = self.categorize(x)
        P = P.cumsum(1)
        S = len(x)
        y = rand.random(S)[:,newaxis] > P # [0,1)
        return y.sum(1)

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

# recursively partitioned BBM
# deals with category-sorted data such that len(Nk) == K
# and len(x) == sum(Nk)
class BBMr:
    def __init__(self, x, Nk=None, verb=False):
        if Nk is None: # one category
            Nk = np.array([len(x)])
        self.x = x
        self.S = x.shape[0]
        self.M = x.shape[1]
        self.verb = verb

        self.recompute(Nk)

    # Recompute handy sufficient-statistics based
    # on this categorization.
    def recompute(self, Nk):
        assert self.S == np.sum(Nk)

        self.Nk = Nk
        self.K = len(Nk)

        # Sufficient statistics for P(theta | xz),
        # Take a set of samples, x : {0,1}^(S*M)
        # and a categorization,  k : {0,..,K-1}^S
        # and compute Nk : int^K, Mj int^(K*M)
        self.Mj = np.zeros((self.K, self.M), int)
        for k, x in enumerate(self.loop()):
            self.Mj[k] = np.sum(x, 0)

    # Return a view into the local x
    def get(self, k):
        start = np.sum(self.Nk[:k])
        return self.x[start:start+self.Nk[k]]

    # loop over every category, returning its x-values
    def loop(self):
        n = 0
        for nk in self.Nk:
            yield self.x[n:n+nk]
            n += nk

    # sample a Bernoulli Mixture distribution from x and k
    # K is the number of categories (should be k.max()+1)
    def sampleBernoulliMixture(self, max_tries = -1):
        alpha = self.M + eta
        rej = 0
        while True:
            B = BernoulliMixture(randBeta(self.Mj+1, self.Nk[:,newaxis]-self.Mj+1), rand.dirichlet(self.Nk+alpha), alpha)
            if rand.uniform() < np.exp(B.logprior()):
                break
            rej += 1
            if rej == max_tries:
                return None
            if rej == 50 and max_tries < 50:
                print("Rejecting lots of samples.")
        if rej >= 50:
            print("Done. %d rejected samples"%rej)
        return B

    ## TODO: create a maximum-likelihood Bernoulli mixture
    #def maxBernoulliMixture(x, k, K):

    # , ,                __            
    # |V| _ ._ ,|_ _    /   _  ._ |  _ 
    # | |(_)| | |_(/_   \__(_| |  | (_)
    #
    # The Monte Carlo trial moves below return True on success
    # or False on failure.  They modify the curren structure
    # in-place.

    # Perform recategorization move by sampling a Bernoulli mixture
    # and returning a new BBMr with new alliances.
    def recategorize(self, max_tries=-1):
        B = self.sampleBernoulliMixture(max_tries)
        if B is None:
            return False
        z = B.classify(self.x)

        y, Nk = reshuffle(self.x, z)

        self.x[:] = y
        self.recompute(Nk)
        return True

    # attempt a split or combine
    def morph(self, split_freq=0.5):
        if self.K == 1 or uniform() < split_freq:
            Q = self.split_choice()
            k, j = choose(Q)
            return self.split(k, j, split_freq, np.sum(Q))

        # prob of choosing this join is (1-split_freq)*2/K*(K-1)
        u = int(uniform()*self.K)
        v = int(uniform()*(self.K-1))
        v += (v>=u)

        Qsum = np.dot(self.Mj[u]+self.Mj[v]+1,
                 self.Nk[u]+self.Nk[v]-self.Mj[u]-self.Mj[v]+1)
        for k in range(self.K-2):
            k += (k>=u) + (k>=v) # skip these two
            Qsum += np.dot(self.Mj[k]+1, self.Nk[k]-self.Mj[k]+1)
        return self.combine(u, v, split_freq, Qsum)

    # Attempt to combine 2 categories.
    def combine(self, u, v, split_freq, Qsum):
        u, v = min(u,v), max(u,v)

        if not accept_join(self.Mj[u], self.Nk[u],
                           self.Mj[v], self.Nk[v],
                           self.S, self.K-1, split_freq, Qsum):
            return False
        if self.verb:
            print("Combined categories %d and %d - %d"%(u,v,self.K-1))
        self.last = ("join", u, v)
        self.join(u, v)
        return True

    def join(self, u, v):
        u, v = min(u,v), max(u,v)
        # Divide into 4 logical categories:
        # ----                    ----
        # 0, ..., u               0, ..., u
        # ----                    ----
        # u+1, ..., u+nl          v
        # ----                    ----
        # v                       u+1, ..., u+nl
        # ----                    ----
        # v+1, ..., K-1           v+1, ..., K-1
        # ----                    ----
        if v != u+1:
            nv = self.Nk[v]
            end = np.cumsum(self.Nk) # end of ea. range

            Nk = self.Nk.tolist()
            Nk[u+1], Nk[v] = Nk[v], Nk[u+1]

            tmp = self.x[end[u]:end[v]].copy()
            end -= end[u]        # index into tmp
            end2 = np.cumsum(Nk) # new range ends

            self.x[end2[u]:end2[u+1]] = tmp[end[v-1]:end[v]]
            nl = v-u-1 # number of categories between u,v
            self.x[end2[u+1]:end2[u+1+nl]] = tmp[:end[v-1]]

            # swap Mj counts
            tmp = self.Mj[v].copy()
            self.Mj[v] = self.Mj[u+1]
            self.Mj[u+1] = tmp

            self.Nk = np.array(Nk)
            v = u+1

        # only have to shift Nk, Mj
        Nk = self.Nk.tolist()
        nv = Nk.pop(v)
        Nk[u] += nv
        self.Nk = np.array(Nk)
        self.K -= 1
        self.Mj[u] += self.Mj[v]
        for i in range(u+1, self.K):
            self.Mj[i] = self.Mj[i+1]
        self.Mj = self.Mj[:-1]

    # attempt a split move
    # pacc = prob. of generating the reverse(combination)
    #          / prob. of getting here
    def split(self, k, j, split_freq, Qsum):
        x = self.get(k)
        p = uniform()
        q = uniform()
        pR = p*x[:,j] + q*(1-x[:,j]) # p or q, depending on bit j
        z = rand.random(pR.shape) < pR # sub-categorization
        z = z.astype(np.int)

        NR = np.sum(z)
        NL = len(x) - NR
        NRj = np.dot(z, x)
        NLj = self.Mj[k] - NRj # dot(1-z, x) = Mj[k]-NRj

        if not accept_split(NLj, NL, NRj, NR,
                            self.S, self.K, split_freq, Qsum):
            return False
        self.last = ("split", k)

        if self.verb:
            print("Split %d on %d - %d"%(k,j, self.K+1))

        L = x[z == 0] # these copy x
        R = x[z == 1]
        x[:len(L)] = L
        x[len(L):] = R

        self.K += 1
        self.Nk = split_row(self.Nk, k, NL, NR)
        self.Mj = split_row(self.Mj, k, NLj, NRj)

        return True

    # Calculate the probability of choosing split (k,j)
    # returns a matrix of un-normalized probabilities.
    def split_choice(self):
        #Q = np.zeros((self.K, self.M))
        #for k, x in enumerate(self.loop()):
            #Q[k] = calc_Qk(x, self.Mj[k])
        Q = (self.Mj+1) * (self.Nk[:,newaxis]-self.Mj+1)
        return Q

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

# permute data according to z and return BBMr class
def mk_BBMr(x, z):
    y, Nk = reshuffle(x, z)
    return BBMr(y, Nk)

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
        if len(best) < B.K or prob > best[B.K-1]:
            if len(best) < B.K:
                best.append(0)
            best[B.K-1] = prob
            print("New best likelihood: %e"%prob)
            print("Members: %s"%B.c)
            out = Path("sz%d"%B.K)
            out.mkdir(exist_ok=True)
            with open(out / "info.txt", "w") as f:
                f.write("# log-likelihood = %e\n"%prob
                        + '\n'.join("%2d %d"%(i,n) for i,n in enumerate(Nk))
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

