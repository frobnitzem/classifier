from classify import *

import numpy as np
newaxis = np.newaxis
rand = np.random.default_rng() # np.random.Generator instance
uniform = rand.random # used for CPU-uniform random numbers

def gen_chomp(N, P, sigma):
    x = np.zeros((N,2*P-1,2))
    # harmonic oscillator snapshots, conc. toward range ends
    t = 0.25*np.pi*(np.cos(np.arange(N).astype(float)*np.pi/N) + 1.0)
    x[:,:P,0] = np.arange(P)
    x[:,P:,0] = np.arange(1,P)*np.sin(t)[:,newaxis]
    x[:,P:,1] = np.arange(1,P)*np.cos(t)[:,newaxis]
    return x + rand.standard_normal(x.shape)*sigma

def gen_heli(N, P, sigma):
    x = np.zeros((N,2*P,3))
    t = np.arange(N).astype(float)*np.pi/N

    x[:,:P,0] = np.arange(P)
    x[:,P+1:,0] = np.cos(t)[:,newaxis]*np.arange(1,P)
    x[:,P+1:,1] = np.sin(t)[:,newaxis]*np.arange(1,P)
    x[:,P:,2] = 1.0
    return x + rand.standard_normal(x.shape)*sigma

def dist_features(x, cut=2.0):
    cut2 = cut*cut

    N = x.shape[1]
    M = N*(N-1) // 2
    y = np.zeros((len(x),M), np.uint8)
    for i,xi in enumerate(x):
        ok = np.sum( (xi[:,newaxis]-xi)**2, 2 ) <= cut2
        k = 0
        for j in range(N-1):
            m = N-1-j
            y[i,k:k+m] = ok[j,j+1:]
            k += m

    return y

def test_combine():
    B = BernoulliMixture(np.array([ [ 0.997,  0.1, 0.3,  0.04, 0.1,   0.1],
                                    [ 0.5,    0.0, 0.1,  0.02, 0.991, 0.8],
                                    [ 0.997,  0.1, 0.3,  0.04, 0.1,   0.1] ]),
                         np.ones(3)/3.0)
    Nk = np.array([3,2,5])
    x = B.sample(Nk)
    BBM = BBMr(x.copy(), Nk)
    #print(x)
    #print(BBM.Nk)
    #print(BBM.Mj)
    Mj = [BBM.Mj[0]+BBM.Mj[2], BBM.Mj[1].copy()]
    while not BBM.combine(0, 2, 0.9, 1.0):
        pass
    #print(BBM.x)
    #print(BBM.Nk)
    #print(BBM.Mj)
    assert np.allclose(np.array([Nk[0]+Nk[2],Nk[1]]), BBM.Nk)
    assert np.allclose(Mj, BBM.Mj)
    assert np.allclose(BBM.x[:Nk[0]],  x[:Nk[0]])
    assert np.allclose(BBM.x[Nk[0]:Nk[0]+Nk[2]], x[-Nk[2]:])
    assert np.allclose(BBM.x[-Nk[1]:], x[Nk[0]:-Nk[2]])

def test_prior_deriv(BBM, x):
    print(BBM.c)
    print(BBM.p)
    from ucgrad import Ndiff
    def plike(p):
        B = BernoulliMixture(p, BBM.c)
        return B.logprior()

    d1 = Ndiff(BBM.p, plike, 1e-9)
    d2 = BBM.d_logprior()
    max_err = np.abs(d1-d2).max()/np.abs(d1).max()
    print("Max derivative error (prior) = %e"%max_err)

def test_deriv(BBM, x):
    from ucgrad import Ndiff
    def plike(p):
        B = BernoulliMixture(p, BBM.c)
        return B.likelihood(x)
    def clike(c):
        B = BernoulliMixture(BBM.p, c)
        return B.likelihood(x)

    d1 = Ndiff(BBM.p, plike, 1e-8)
    d2 = Ndiff(BBM.c, clike, 1e-8)
    d3, d4 = BBM.d_like(x)
    max_err = np.abs(d1-d3).max()/np.abs(d1).max()
    print("Max derivative error (p) = %e"%max_err)
    max_err = np.abs(d2-d4).max()/np.abs(d2).max()
    print("Max derivative error (c) = %e"%max_err)

def test_sample(x, samples=100):
    y = dist_features(x)
    x = compress_features(y)

    BBM = BBMr(x.copy())
    acc = 0
    for i in range(samples):
        BBM.recategorize()
        acc += BBM.morph()
        #if (i+1)%10 == 0:
        #    print("  Sample %d, %d moves accepted."%(i+1,acc))

    print("%d of %d moves accepted"%(acc,i+1))
    print(BBM.Nk)
    print(BBM.Mj)
    B = BBM.sampleBernoulliMixture()
    z = B.classify(x)
    print(z)

def test_features():
    x = gen_chomp(4, 4, 0.0)
    y = dist_features(x)
    #ham = np.sum(y[:,newaxis] ^ y, 2) # pairwise Hamming (L1) distance
    x = compress_features(y)
    BBM = BBMr(x.copy(), np.array([4]))
    assert BBM.M == 8
    Mj = np.array([[3, 1, 3, 3, 1, 1, 1, 1]])
    assert np.allclose(BBM.Mj, Mj)
    assert np.allclose(BBM.get(0), x)
    assert BBM.recategorize()
    assert np.allclose(BBM.x, x)

    Q = BBM.split_choice()
    Qsum = np.sum(Q)
    assert np.allclose(Q, 8)
    assert Qsum == 64

    k, j = 0, 6
    while not BBM.split(k, j, 0.5, Qsum):
        pass
    print(BBM.Nk)
    print(BBM.Mj)
    assert BBM.K == 2
    assert np.sum(BBM.Nk) == len(x)
    assert np.allclose( np.sum(BBM.Mj,0), Mj )

    #Qsum = np.dot(Mj+1, len(x)-Mj+1) # same as earlier Qsum
    while not BBM.combine(0, 1, 0.5, Qsum):
        pass

    print("combined!")
    print(BBM.Nk)
    print(BBM.Mj)
    assert BBM.K == 1
    assert np.allclose(BBM.Mj, Mj)
    assert np.allclose(BBM.Nk, np.array([4]))
 
def test_bbm():
    B = BernoulliMixture(np.array([ [ 0.0001, 0.5,    0.98, 0.02, 0.5,   0.999999],
                                    [ 0.5,    0.0001, 0.1,  0.02, 0.991, 0.8],
                                    [ 0.997,  0.1,    0.3,  0.04, 0.1,   0.1] ]),
                         np.ones(3)/3.0)
    #print(B.prior(True))
    N = np.array([3000,2000,5000]) # 10k samples
    x = B.sample(N)

    BBM = BBMr(x, N)
    print(BBM.Mj)
    print((B.p*np.array(N)[:,newaxis]).astype(np.int))

    #cat = B.classify(x)
    #print(cat)
    z = B.categorize(x)

    return B, x

if __name__=="__main__":
    #test_features()
    #x = gen_chomp(1000, 40, 0.2)
    #x = gen_heli(1000, 40, 0.2)
    #test_sample(x, 1000)
    #test_combine()
    B, x = test_bbm()
    test_prior_deriv(B, x)
    test_deriv(B, x)

