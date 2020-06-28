# Helper function for probability-related concerns

import numpy as np
newaxis = np.newaxis
rand = np.random.default_rng() # np.random.Generator instance
uniform = rand.random # used for CPU-uniform random numbers

from scipy.special import gammaln

def compress_features(x, verb=True):
    Mi = np.sum(x,0)
    zeros = Mi == 0
    ones  = Mi == len(x)
    if verb:
        print("%d are always zero and %d are always one"%(np.sum(zeros), np.sum(ones)))
    ones += zeros
    ind = [ i for i in range(x.shape[1]) if not ones[i] ]
    if verb:
        print(Mi[ind])
    return x[:,ind] # truncate feature space to the interesting subset

def load_features(name):
    x = np.load(name)[::10]
    x = np.unpackbits( x ).reshape( (len(x),-1) )
    print("Input is %s samples x %s features" % x.shape)
    return compress_features(x)

def choose(p):
    sh = p.shape
    p = np.cumsum( np.reshape(p, -1) )
    i = np.sum(uniform()*p[-1] > p)
    if len(sh) == 1:
        return i
    idx = []
    for N in sh:
        idx.append(i % N)
        i = i // N
    return tuple(idx)

# probabilities are exp(Q)
def choose_exp(Q):
    M = Q.max()
    return choose(np.exp(Q-M))

# Draw N samples from the Bernoulli distribution
def Bernoulli(p, N):
    M = len(p)
    return rand.random((N,M)) < p

# modified randBeta dealing with alpha, beta = 0
def randBeta(alpha, beta):
    m0 = alpha == 0 # p must be 0
    m1 = beta == 0  # p must be 1
    p = rand.beta(alpha+m0, beta+m1)
    p += (1.0-p)*m1 # set to one here
    p -= p*m0       # set to zero here
    return p

# permute data according to z
def reshuffle(x, z):
    K = z.max()+1
    Nk = np.array( [np.sum(z==k) for k in range(K)] )
    # nonzero categories
    idx = [k for k,nk in enumerate(Nk) if nk > 0]

    y = np.zeros(x.shape, x.dtype)
    n = 0
    for k in idx:
        y[n:n+Nk[k]] = x[ z==k ]
        n += Nk[k]
    return x, Nk[idx]

# Split row k of M in two, inserting a and b in its place.
def split_row(M, k, a, b):
    A = np.empty((len(M)+1,)+M.shape[1:], M.dtype)
    A[:k]  = M[:k]
    A[k]   = a
    A[k+1] = b
    A[k+2:] = M[k+1:]
    return A

# Calculate the log-likelihood for splitting
# a category into L and R
def calc_Qxy(NLj, NL, NRj, NR, N, K):
    M = len(NLj)
    assert M == len(NRj)

    lp = np.sum( Ginf(NLj, NL, 2) + Ginf(NRj, NR, 2) \
                - Ginf(NLj+NRj, NL+NR, 2) )
    return lp + Ginf(NL, NL+NR) - np.log(N*(N+K))

# Calculate the log-probability of generating this
# particular L,R split.  Any j could be used,
# so we need the sum.
# FIXME: reconsider prob. of NL,NR == 0 or of sum_j Q_{kj} vs. sum_{kj} Q_{kj}
def calc_Qgen(NLj, NL, NRj, NR):
    M = len(NLj)
    assert M == len(NRj)

    NKj = NLj+NRj
    NK  = NL+NR
    gen = Ginf(NLj, NL) + Ginf(NRj, NR) - Ginf(NKj, NK) \
          + Ginf(NL, NK)
    # Generation prob. is proportional to this factor:
    #   - np.log((Nk+1)*(N-Nk+1))
    # so we omit it here, since it cancelled.

    lp  = gen.max() # log( sum(exp(gen)) )
    lp += np.log( np.sum(np.exp(gen-lp)) ) # in [0, log(M)]
    return lp

# K -- corresponds to before split
def split_lp(NLj, NL, NRj, NR, N, K, split_freq, Qsum):
    freq = split_freq + (1.0 - split_freq)*(K == 1)
    pacc = (1.0-split_freq)*2*Qsum / (freq*K*(K+1))

    lp = np.log(pacc) - calc_Qgen(NLj, NL, NRj, NR)
    lp += calc_Qxy(NLj, NL, NRj, NR, N, K)
    #print("%e %e %e %e"%(np.log(pacc), calc_Qgen(NLj, NL, NRj, NR), calc_Qxy(NLj, NL, NRj, NR, N, K), lp))
    return lp

# K -- corresponds to before split
def accept_split(NLj, NL, NRj, NR, N, K, split_freq, Qsum):
    if NL == 0 or NR == 0:
        return False
    lp = split_lp(NLj, NL, NRj, NR, N, K, split_freq, Qsum)
    return lp >= 0.0 or np.exp(lp) > uniform()

# K -- corresponds to before split
def accept_join(NLj, NL, NRj, NR, N, K, split_freq, Qsum):
    lp = -split_lp(NLj, NL, NRj, NR, N, K, split_freq, Qsum)
    return lp >= 0.0 or np.exp(lp) > uniform()

# Calculate the split log-likelihood estimate
# for the category (returns vector over j)
def calc_Qk(x, Mk=None, alpha=0.9):
    M = x.shape[1]
    if Mk is None:
        Mk = np.sum(x,0)

    blksz = 160 # block of features
    Dnorm = np.sum( Ginf(Mk, len(x), 2) )
    #Dnorm += np.log(K*(N+K)) # cost for creating an add'l category in the first place

    ans = []
    for i in range(0, x.shape[1], blksz):
        sl = slice(i, min(i+blksz, x.shape[1]))
        y = x[:,sl].transpose().astype(np.uint32)
        Mkj = np.dot(y, x)

        NR = Mk[sl,newaxis]
        NRj = Mkj
        NL = len(x)-NR
        NLj = Mk[newaxis,:]-Mkj # Mj - Mkj

        DR = np.sum( Ginf(alpha*NRj + (1-alpha)*NLj, alpha*NR+(1-alpha)*NL), 1)
        DL = np.sum( Ginf(alpha*NLj + (1-alpha)*NRj, alpha*NL+(1-alpha)*NR), 1)
        D0 = Ginf(alpha*Mk[sl]+(1-alpha)*(len(x)-Mk[sl]), len(x))

        ans.extend( (DL + DR - D0 - Dnorm).tolist() )

    return np.array(ans)

def Ginf(a, b, n2=1):
    return gammaln(a+1)+gammaln(b-a+1)-gammaln(b+n2)

