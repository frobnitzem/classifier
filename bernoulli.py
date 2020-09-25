from prob import *

#rand = np.random.default_rng(seed) # np.random.Generator instance
#uniform = rand.random # used for CPU-uniform random numbers

# Mixture of independent multivariate Bernoulli distributions.
class BernoulliMixture:
    def __init__(self, p, c, alpha, rand):
        assert len(p.shape) == 2 and len(c.shape) == 1
        self.K = len(c)
        assert len(p) == self.K
        self.M = p.shape[1]
        self.alpha = alpha
        self.rand = rand

        self.p = p
        self.c = c

    # Calc. 1 - Bhattacharyya similarity to all p
    def bdist(self, v):
        p = self.p
        B = np.prod( np.sqrt(p*v[newaxis,:])
                    + np.sqrt((1-p)*(1-v)[newaxis,:]),
                    1)
        return 1-B

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
        P1 = np.log(self.p*x[:,newaxis,:] + (1-self.p)*(1-x[:,newaxis,:])).sum(2) + np.log(self.c) # S x K
        Pm = P1.max(1) # max cat. term for each sample
        P1 -= Pm[:,newaxis]
        Pm += np.log(np.exp(P1).sum(1))
        lP = Pm.sum() + self.logprior() + cat_prior(len(x), self.alpha, self.M, self.K) \
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
            x[k:k+Nk] = Bernoulli(pk, Nk, self.rand)
            k += Nk
        return x

    def categorize(self, x):
        # compute P(k | x, p, c) : S by K
        # note - this only works for x = 0 or 1
        # x is S by M
        # p is K by M
        P = np.log(self.p*x[:,newaxis,:] + (1.0-self.p)*(1-x[:,newaxis,:])).sum(2) + np.log(self.c)
        P = np.exp(P - P.max(1)[:,newaxis])
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
        y = self.rand.random(S)[:,newaxis] > P # [0,1)
        return y.sum(1)

# Create a Bernoulli Mixture from a list of categories
def mkBern(cat, rand, max_tries = 0):
    Nk = np.array( [c.N for c in cat] )
    rej = 0
    while True:
        mu = np.array( [c.mu(rand) for c in cat] )
        B = BernoulliMixture(mu, rand.dirichlet(Nk+alpha), alpha, rand)
        if rand.random() < np.exp(B.logprior()):
            break
        rej += 1
        if rej == max_tries:
            return None
        if rej == 50 and max_tries > 50:
            print("Rejecting lots of samples.")
    if rej >= 50 and max_tries > 50:
        print("Done. %d rejected samples"%rej)

    return B

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

    def mu(self, rand): # sample mu from the default dist'n
        return randBeta(self.Mj+1, self.N-self.Mj+1, rand)

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

def bdist(u, v):
    return 1 - np.prod( np.sqrt(u*v) + np.sqrt((1-u)*(1-v)) )

# Recursively partitioned BBM.
# Deals with category-sorted data such that len(Nk) == K
# and len(x) == sum(Nk)
class BBMr:
    def __init__(self, x, rand, Nk=None, verb=False):
        if Nk is None: # one category
            Nk = np.array([len(x)])
        self.x = x
        self.rand = rand
        self.S = x.shape[0]
        self.M = x.shape[1]
        self.verb = verb

        self.alpha = self.M+1
        self.recompute(Nk)

    # Recompute handy sufficient-statistics based
    # on this categorization.
    def recompute(self, Nk):
        assert self.S == np.sum(Nk)

        self.Nk = np.array(Nk) # also copies
        self.K = len(Nk)

        # Sufficient statistics for P(theta | xz),
        # Take a set of samples, x : {0,1}^(S*M)
        # and a categorization,  k : {0,..,K-1}^S
        # and compute Nk : int^K, Mj int^(K*M)
        self.Mj = np.zeros((self.K, self.M), int)
        for k, x in enumerate(self.loop()):
            self.Mj[k] = np.sum(x, 0)

        self.B = sampleBernoulliMixture(self.Nk, self.Mj, self.alpha, self.rand)

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

    ## TODO: create a maximum-likelihood Bernoulli mixture
    #def maxBernoulliMixture(x, k, K):

    # , ,                __            
    # |V| _ ._ ,|_ _    /   _  ._ |  _ 
    # | |(_)| | |_(/_   \__(_| |  | (_)
    #
    # The Monte Carlo trial moves below return True on success
    # or False on failure.  They modify the current structure
    # in-place.

    # Perform recategorization move by sampling a Bernoulli mixture
    # and returning a new BBMr with new alliances.
    def recategorize(self):
        z = self.B.sample_k(self.x)
        y, Nk = reshuffle(self.x, z)
        assert len(z) == self.S
        assert Nk.sum() == self.S
        if len(Nk) != self.K:
            return False

        Mj = calc_Mj(y, Nk)
        B = sampleBernoulliMixture(Nk, Mj, self.alpha, self.rand, 50)
        if B is None:
            return False
        self.x[:] = y
        self.B = B
        self.Nk = Nk
        self.Mj = Mj
        self.K = len(Nk)
        return True

    def accept_split(self, k, L, R, split_freq, beta, Qsum):
        if L.N == 0 or R.N == 0:
            return False
        muL = L.mu(self.rand)
        muR = R.mu(self.rand)
        DL = self.B.bdist(muL)
        DL[k] = 1.0
        DR = self.B.bdist(muR)
        DR[k] = 1.0
        Dk = self.B.bdist(self.B.p[k])
        Dk[k] = 1.0

        lp = split_prefactor(L.Mj, L.N, R.Mj, R.N, self.S, self.K, split_freq, beta, Qsum)
        lp += np.log(DL).sum() + np.log(DR).sum() - np.log(Dk).sum() \
            + np.log(bdist(muL, muR))
        if not metropolis(lp, self.rand):
            return None
        return muL, muR

    def accept_join(self, u, v, split_freq, beta, Qsum):
        L = Category(self.Mj[u], self.Nk[u])
        R = Category(self.Mj[v], self.Nk[v])

        mu = (L+R).mu(self.rand)
        DL = self.B.bdist(self.B.p[u])
        DL[u] = 1.0
        DL[v] = 1.0
        DR = self.B.bdist(self.B.p[v]) # count the u-v distance once
        DR[v] = 1.0

        Dk = self.B.bdist(mu)
        Dk[u] = 1.0
        Dk[v] = 1.0

        lp = split_prefactor(L.Mj, L.N, R.Mj, R.N, self.S, self.K-1, split_freq, beta, Qsum)
        lp += np.log(DL).sum() + np.log(DR).sum() - np.log(Dk).sum()
        if not metropolis(-lp, self.rand):
            return None
        return mu

    # attempt a split or combine
    def morph(self, split_freq=0.5, beta=0.9):
        uniform = self.rand.random
        if self.K == 1 or uniform() < split_freq:
            Q = self.split_choice()
            k, j = choose(Q, self.rand)
            return self.split(k, j, split_freq, beta, np.sum(Q))

        # prob of choosing this join is (1-split_freq)*2/K*(K-1)
        u = int(uniform()*self.K)
        v = int(uniform()*(self.K-1))
        v += (v>=u)

        # would-be eligible splits
        Q2 = self.split_choice()
        Q2[u] = 0
        Q2[v] = (self.Nk[u]+self.Nk[v] != self.Mj[u]+self.Mj[v]) \
              * (self.Mj[u]+self.Mj[v] != 0)
        return self.combine(u, v, split_freq, beta, Q2.sum())

    # Attempt to combine 2 categories.
    def combine(self, u, v, split_freq, beta, Qsum):
        u, v = min(u,v), max(u,v)

        mu = self.accept_join(u, v, split_freq, beta, Qsum)
        if mu is None:
            return False
        if self.verb:
            print("Combined categories %d and %d - %d"%(u,v,self.K-1))
        self.join(u, v, mu)
        return True

    # Combine categories u and v, with mu representing the resulting category
    def join(self, u, v, mu):
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
            #tmp = self.Mj[v].copy()
            #self.Mj[v] = self.Mj[u+1]
            #self.Mj[u+1] = tmp

            #self.Nk = np.array(Nk)
            #v = u+1

        # only have to shift Nk, Mj, B.p
        self.K -= 1
        self.Nk[u] += self.Nk[v]
        self.Nk = del_row(self.Nk, v)
        self.Mj[u] += self.Mj[v]
        self.Mj = del_row(self.Mj, v)

        p = del_row(self.B.p, v)
        p[u] = mu
        c = self.rand.dirichlet(self.Nk+self.alpha)
        self.B = BernoulliMixture(p, c, self.alpha, self.rand)

    # attempt a split move
    # pacc = prob. of generating the reverse(combination)
    #          / prob. of getting here
    def split(self, k, j, split_freq, beta, Qsum):
        x = self.get(k)
        NR = 0
        if self.verb:
            assert len(x) == self.Nk[k]
            assert len(x) > 2
            assert self.Mj[k,j] != self.Nk[k]
            assert self.Mj[k,j] != 0
        while NR == 0 or NR == self.Nk[k]:
            pR = beta*x[:,j] + (1-beta)*(1-x[:,j])
            z = self.rand.random(pR.shape) < pR # sub-categorization
            z = z.astype(np.int)
            NR = np.sum(z)

        NL = len(x) - NR
        NRj = np.dot(z, x)
        NLj = self.Mj[k] - NRj # dot(1-z, x) = Mj[k]-NRj

        L = Category(NLj, NL)
        R = Category(NRj, NR)
        ret = self.accept_split(k, L, R, split_freq, beta, Qsum)
        if ret is None:
            return False
        muL, muR = ret

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
        self.B = BernoulliMixture(split_row(self.B.p, k, muL, muR),
                                  self.rand.dirichlet(self.Nk+self.alpha), self.alpha, self.rand)

        return True

    # Calculate the probability of choosing split (k,j)
    # returns a matrix of un-normalized probabilities.
    def split_choice(self):
        return ( (self.Nk[:,newaxis] != self.Mj)*(self.Mj != 0) ).astype(float)

def calc_Mj(x, Nk):
    K = len(Nk)
    M = x.shape[1]
    Mj = np.zeros((K, M), int)
    n = 0
    for k, nk in enumerate(Nk):
        Mj[k] = x[n:n+nk].sum(0)
        n += nk
    return Mj

# sample a Bernoulli Mixture distribution from x and k
# K is the number of categories (should be k.max()+1)
def sampleBernoulliMixture(Nk, Mj, alpha, rand, max_tries=-1):
    K = len(Nk)
    assert K == Mj.shape[0]
    rej = 0
    while True:
        B = BernoulliMixture(randBeta(Mj+1, Nk[:,newaxis]-Mj+1, rand),
                             rand.dirichlet(Nk+alpha), alpha, rand)
        if metropolis(B.logprior(), rand):
            break
        rej += 1
        if rej == max_tries:
            return None
        if rej == 50 and max_tries < 50:
            print("Rejecting lots of samples.")
    if rej >= 50:
        print("Done. %d rejected samples"%rej)
    return B

# permute data according to z and return BBMr class
def mk_BBMr(x, z):
    y, Nk = reshuffle(x, z)
    return BBMr(y, Nk)

