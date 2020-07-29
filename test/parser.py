#!/usr/bin/env python3

import sys
import numpy as np
from test_helpers import sequential

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

acc, acc2 = sequential(np.array([int(k) for k in nlist]))
print("%.1f %.1f"%(acc, acc2))

