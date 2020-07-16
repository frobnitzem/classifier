# Helper function for probability-related concerns

eta = 1

import numpy as np
newaxis = np.newaxis
rand = np.random.default_rng() # np.random.Generator instance
uniform = rand.random # used for CPU-uniform random numbers
np.seterr(under='ignore') # exp(-800) == 0 is OK

from scipy.special import gammaln

def compress_features(x, verb=True):
    Mi = np.sum(x,0)
    zeros = Mi == 0
    ones  = Mi == len(x)
    if verb:
        print("%d all-zero and %d all-one features"%(np.sum(zeros), np.sum(ones)))
    ones += zeros
    ind = [ i for i in range(x.shape[1]) if not ones[i] ]
    if verb:
        print(Mi[ind])
    return np.array(ind), x[:,ind] # truncate feature space to the interesting subset

def load_features(name, ind=None, sl=None):
    x = np.load(name)
    if sl is not None:
        x = x[sl]
    x = np.unpackbits( x ).reshape( (len(x),-1) )
    print("Input is %s samples x %s features" % x.shape)
    if ind is None:
        return compress_features(x)
    return ind, x[:,ind]

def choose(p):
    sh = p.shape
    p = np.cumsum( np.reshape(p, -1) )
    i = np.sum(uniform()*p[-1] > p)
    if len(sh) == 1:
        return i
    idx = []
    for N in sh[::-1]:
        idx.append(i % N)
        i = i // N
    return tuple(idx[::-1])

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

# if mask == False, then it subtracts denominator of log integral{\pi^{Nk+\alpha+1}}
#                      (i.e. gammaln(N + alpha*K))
def cat_prior(N,alpha,M,K,mask=True):
    # simplistic model
    return gammaln(alpha*K)-K*gammaln(alpha) - (1-mask)*gammaln(N + alpha*K)
    #eta = alpha - M
    # model swapping overall normalization for fixed const. (pretends like alpha = eta)
    #return mask*gammaln(N + alpha*K) - gammaln(K+1) - gammaln(N + eta*K) # mask == True
    #return mask*gammaln(N + alpha*K) - gammaln(N + eta*K) # mask == True

# Calculate the log-likelihood for splitting
# a category into L and R
def calc_Qxy(NLj, NL, NRj, NR, N, K):
    M = len(NLj)
    assert M == len(NRj)

    alpha = M+eta
    lp = np.sum(Ginf(NLj, NL, 2) + Ginf(NRj, NR, 2) - Ginf(NLj+NRj, NL+NR, 2))
    lp += cat_prior(N,alpha,M,K+1,False) - cat_prior(N,alpha,M,K,False)
    return lp + gammaln(NL+alpha) + gammaln(NR+alpha) - gammaln(NL+NR+alpha)

# Calculate the log-probability of generating this
# particular L,R split.  Any j could be used, so we need the sum.
def calc_Qgen(NLj, NL, NRj, NR, beta=0.9):
    M = len(NLj)
    assert M == len(NRj)

    lb = np.log(beta)
    nlb = np.log(1.0-beta)

    NKj = NLj+NRj
    NK  = NL+NR
    mask = ((NKj != NK) * (NKj != 0)).astype(int)

    genL = (NLj+NR-NRj)*lb + (NRj+NL-NLj)*nlb
    genR = (NRj+NL-NLj)*lb + (NLj+NR-NRj)*nlb
    # P = beta^(correct cat.) (1-beta)^(incorrect cat.) / 1.0 - P(all L) - P(all R)
    #
    den = np.log(1.0
            - np.exp(NKj*lb+(NK-NKj)*nlb)
            - np.exp(NKj*nlb+(NK-NKj)*lb) )
    genL -= den
    genR -= den

    # Generation prob. is proportional to this factor:
    #   - np.log((Nk+1)*(N-Nk+1))
    # so we omit it here, since it cancelled.

    lp = max(genL.max(), genR.max()) # log( sum(exp(gen)) )
    #if np.isinf(lp):
    #    raise FloatingPointError

    lp += np.log( 0.5*(np.dot(np.exp(genL-lp), mask)
                     + np.dot(np.exp(genR-lp), mask))
                ) # in [0, log(M)]
    return lp

# K -- corresponds to before split
def split_lp(NLj, NL, NRj, NR, N, K, split_freq, beta, Qsum):
    freq = split_freq + (1.0 - split_freq)*(K == 1)
    pacc = (1.0-split_freq)*2*Qsum / (freq*K*(K+1))

    lp = np.log(pacc) - calc_Qgen(NLj, NL, NRj, NR, beta)
    lp += calc_Qxy(NLj, NL, NRj, NR, N, K)
    #print("%e %e %e %e"%(np.log(pacc), calc_Qgen(NLj, NL, NRj, NR, beta), calc_Qxy(NLj, NL, NRj, NR, N, K), lp))
    return lp

# K -- corresponds to before split
def accept_split(NLj, NL, NRj, NR, N, K, split_freq, beta, Qsum):
    if NL == 0 or NR == 0:
        return False
    lp = split_lp(NLj, NL, NRj, NR, N, K, split_freq, beta, Qsum)
    return lp >= 0.0 or np.exp(lp) > uniform()

# K -- corresponds to before split
def accept_join(NLj, NL, NRj, NR, N, K, split_freq, beta, Qsum):
    lp = -split_lp(NLj, NL, NRj, NR, N, K, split_freq, beta, Qsum)
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
        y = x[:,sl].transpose().astype(np.uint64)
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

