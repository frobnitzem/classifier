# Selections specifically for MPro

import numpy as np
newaxis = np.newaxis

def dist2(x, y, L):
    dr = x[:,newaxis]-y
    dr -= L[2]*np.floor(dr[:,:,2]/L[2,2]+0.5)[:,:,newaxis]
    dr -= L[1]*np.floor(dr[:,:,1]/L[1,1]+0.5)[:,:,newaxis]
    dr -= L[0]*np.floor(dr[:,:,0]/L[0,0]+0.5)[:,:,newaxis]
    return np.sum(dr*dr, 2)

class DistXY:
    def __init__(self, sx, sy, r):
        assert len(sx & sy) == 0
        self.sx = list(sx)
        self.sx.sort()
        self.sy = list(sy)
        self.sy.sort()

        self.R2 = r*r
        self.N = len(sx)*len(sy)

    def feat(self, L, crd):
        x = crd[self.sx]
        y = crd[self.sy]
        r2 = dist2(x, y, L) < self.R2
        for row in r2:
            yield row

    def __str__(self):
        return "XY pairs within %g: %s, %s"%(
                    self.R2**0.5, str(self.sx), str(self.sy))

class DistXX:
    def __init__(self, sel, r):
        self.sel = list(sel)
        self.sel.sort()

        self.R2 = r*r
        self.N = len(sel)*(len(sel)-1)//2

    def feat(self, L, crd):
        x = crd[self.sel]
        r2 = dist2(x, x, L) < self.R2
        for i, row in enumerate(r2[:-1]):
            yield row[i+1:]

    def __str__(self):
        return "XX pairs within %g: %s"%(self.R2**0.5, str(self.sel))

class Sel(list):
    def __init__(self, l):
        list.__init__(self, l)
        self.N = sum(s.N for s in l)

    def feat(self, L, crd):
        for s in self:
            for f in s.feat(L, crd):
                yield f

    def __str__(self):
        desc = "Feature List:\n"
        for s in self:
            desc += " * %s\n" % str(s)
        return desc

def selections(pdb):
    site = pdb.resnum(41) | pdb.resnum(145) | pdb.resnum(188)
    site = site - pdb.chain('B')

    ca = pdb.atname('CA')
    ca_B = ca & pdb.chain('B')
    ca_A = ca - ca_B
    
    site = site - ca_A
    sel = Sel([  DistXX(site, 3),       # site internal conf.
                 DistXX(ca_A, 5),       # A internal conf.
                 DistXY(site, ca_A, 4), # A wrt. site
                 DistXY(ca_A, ca_B, 5)  # A wrt. B
               ])
    print(sel)

    return sel.N, sel.feat

