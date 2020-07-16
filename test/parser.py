#!/usr/bin/env python3

import sys
import numpy as np

fname = sys.argv[1]

nlist = []
state = 0
with open(fname) as f:
    for line in f:
        if state == 0 and line[:14] == "Classification":
            state = 1
            continue
        if state == 1 and '[' in line:
            state = 2
        if state == 2:
            line = line.replace('[', '').replace(']', '')
            nlist.extend(line.strip().split())

if state == 0:
    print("No category list found.")
    exit(1)
if len(nlist) == 0:
    print("Empty category list!")
    exit(2)

misses = 0

nlist = [int(k) for k in nlist]

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

N = len(seen)
misses -= N-1 # first switching point doesn't count
acc = 100 - (misses * 100 / len(nlist)) # accuracy rate
#print(trans)

# re-number to sequential integers
num = {}
for i,n in enumerate(sorted(seen.keys())):
    num[n] = i

T = np.zeros((N,N), np.int)
for (i,j),k in trans.items():
    i,j = num[i], num[j]
    T[i,j] = k

# Step through the trs. matrix, moving to the most connected
# cluster and zeroing out those transitions at ea. step.
i = 0
visited = [i]
while len(visited) < N:
    j = np.argmax(T[i])
    T[i,j] = 0
    i = j
    visited.append(j)

#print(visited)
acc2 = 100 - (T.sum() * 100 / len(nlist)) # accuracy rate (measure 2)
print("%.1f %.1f"%(acc, acc2))

