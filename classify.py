#!/usr/bin/env python3

from prob import *

#import cupy as cp
#rand = cp.random

cat_penalty = 10.0 # log-probability cost per add'l category

# Mixture of independent multivariate Bernoulli distributions.
class BernoulliMixture:
    def __init__(self, p, c):
        assert len(p.shape) == 2 and len(c.shape) == 1
        self.K = len(c)
        assert len(p) == self.K
        self.M = p.shape[1]

        self.p = p
        self.c = c

    def prior(self, verb=False):
        # Calculate a prior probability over categories from a
        # matrix of Bhattacharyya distances
        p = self.p
        pd = np.prod( np.sqrt(p[:,newaxis,:]*p[newaxis,:,:])
                    + np.sqrt((1-p)[:,newaxis,:]*(1-p)[newaxis,:,:]),
                    2)
        if verb:
            print(pd)
        pd -= np.identity(self.K)
        # All K choose 2 categories differ:
        return np.sqrt(np.prod(1-pd))

    # Log-likelihood of a sample, x, given this model
    def likelihood(self, x):
        P = np.dot(np.prod( self.p*x[:,newaxis,:] + (1-self.p)*(1-x[:,newaxis,:]) , 2 ), self.c)
        lP = np.sum(np.log(P)) + np.log(self.prior()) - cat_penalty*self.K
               # - np.log(self.p).sum() - np.log(1 - self.p).sum() - np.dot(self.c, np.log(self.c))
        return lP

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

    # TODO: calculate the maximum likelihood p

    # pull a category sample
    def sample_k(self, x):
        P = self.categorize(x)
        P = P.cumsum(1)
        S = len(x)
        y = rand.random(S)[:,newaxis] > P # [0,1)
        return y.sum(1)


# Old attempt at adding elements to categories sequentially
class BBMs:
    def __init__(self, x, Nk, min_lk=0.9):
        S = x.shape[0]
        M = x.shape[1]
        Nk = np.array( [ 3.0 ] )
        Mj = np.array( [ 1.0 + x[0] ] )
        #B = BernoulliMixture(Mj/Nk[:,newaxis], Nk/sum(Nk))
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
    def __init__(self, x, Nk=None):
        if Nk is None: # one category
            Nk = np.array([len(x)])
        self.x = x
        self.S = x.shape[0]
        self.M = x.shape[1]

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
    def sampleBernoulliMixture(self):
        rej = 0
        while True:
            B = BernoulliMixture(randBeta(self.Mj+1, self.Nk[:,newaxis]-self.Mj+1), rand.dirichlet(self.Nk+1))
            #B = BernoulliMixture(randBeta(self.Mj+1, self.Nk[:,newaxis]-self.Mj+1), rand.dirichlet(self.Nk+0.01*self.S))
            if rand.uniform() < B.prior():
                break
            rej += 1
            if rej == 10:
                print("Rejecting lots of samples.")
        if rej >= 10:
            print("Done, %d rejected samples"%rej)
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
    #
    # Always succeeds.
    def recategorize(self):
        B = self.sampleBernoulliMixture()
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
        print("Combined categories %d and %d - %d"%(u,v,self.K-1))

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

        return True

    # attempt a split move
    # pacc = prob. of generating the reverse(combination)
    #          / prob. of getting here
    def split(self, k, j, split_freq, Qsum):
        x = self.get(k)
        p = uniform()
        q = uniform()
        pR = p*x[:,j] + q*(1-x[:,j]) # p or q, depending on bit j
        z = rand.random() < pR # sub-categorization
        z = z.astype(np.uint64)

        NR = np.sum(z)
        NL = len(x) - NR
        NRj = np.dot(z, x)
        NLj = self.Mj[k] - NRj # dot(1-z, x) = Mj[k]-NRj

        if not accept_split(NLj, NL, NRj, NR,
                            self.S, self.K, split_freq, Qsum):
            return False

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

    # Calculate the log-probability of generating this particular L,R split
    # for this particular k,j
    def calc_Qgen_j(self, x, y, NL=None, NR=None):
        return 1

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
        B = BernoulliMixture(rand.random((K,M)), np.ones(K)/float(K))
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
    print(B.prior(True)) # inter-category distances
    print(BBM.Mj)

    return B, BBM

def main(argv):
    x = load_features(argv[1])

    BBM = BBMr(x)
    acc = 0
    for i in range(10*1000):
        BBM.recategorize()
        acc += BBM.morph()

        if (i+1)%1000 == 0:
            print("Saving...")
            np.save("members.npy", BBM.Nk)
            np.save("features.npy", BBM.Mj.astype(float) / BBM.Nk[:,newaxis])

    print("%d of %d moves accepted"%(acc,i+1))
    print(BBM.Nk)
    print(BBM.Mj)

    np.save("members.npy", BBM.Nk)
    np.save("features.npy", BBM.Mj.astype(float) / BBM.Nk[:,newaxis])

    B = BBM.sampleBernoulliMixture()
    del BBM

    x = load_features(argv[1])
    z = B.classify(x)
    np.save("categories.npy", z)

if __name__=="__main__":
    import sys
    main(sys.argv)

