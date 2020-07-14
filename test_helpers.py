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

