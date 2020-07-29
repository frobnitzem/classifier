import numpy as np
import os
newaxis = np.newaxis
cwd = os.path.dirname(os.path.abspath(__file__))

def read_rule(K):
    assert K > 1
    r = np.fromfile("%s/rules/lebedev_%03d.txt"%(cwd,K), sep=' ').reshape((-1,3))
    r[:,:2] *= np.pi/180.0 # th, phi

    N = len(r)
    x = np.zeros((N,3)) # cos(th) sin(phi), sin(th) sin(phi), cos(phi)
    x[:,0] = np.cos(r[:,0])
    x[:,1] = np.sin(r[:,0])
    x[:,:2] *= np.sin(r[:,1])[:,newaxis]
    x[:,2] = np.cos(r[:,1])
    #w = r[:,-1]*4*np.pi
    return x

# Count percent of sequential categories
# nlist = z, category assigned to each frame
def sequential(nlist):
    misses = 0

    cur = nlist[0]
    seen = { cur: 0 }
    trans = {} # (i,j) : n -- where i < j
    for n in nlist:
        if cur != n:
            ij = min(cur,n), max(cur,n)
            misses += 1
            cur = n
            if cur not in seen:
                seen[cur] = 0

            if ij not in trans:
                trans[ij] = 1
            else:
                trans[ij] += 1
        seen[cur] += 1

    K = len(seen)
    misses -= K-1 # first switching point doesn't count
    acc = 100 - (misses * 100 / len(nlist)) # accuracy rate
    #print(trans)

    # re-number to sequential integers, same sequence order
    num = {}
    for i,n in enumerate(sorted(seen.keys())):
        num[n] = i

    T = np.zeros((K,K), np.int)
    for (i,j),k in trans.items():
        i,j = num[i], num[j]
        T[i,j] = k

    # Step through the trs. matrix, moving to the most connected
    # next step and zeroing out those transitions at ea. step.
    i = num[ nlist[0] ]
    visited = [i]
    while len(visited) < K:
        j = np.argmax(T[i])
        T[i,j] = 0
        i = j
        visited.append(j)

    #print(visited)
    acc2 = 100 - (T.sum() * 100 / len(nlist)) # accuracy rate (measure 2)
    return acc, acc2

