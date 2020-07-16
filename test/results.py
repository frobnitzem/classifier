import numpy as np
import pylab as plt

data = dict(
chomp = [
 [0, 6, 21, 48, 22, 2, 1],
 [0, 1, 24, 39, 28, 8],
 [0, 1, 12, 27, 34, 26],
 [0, 4, 18, 30, 38, 9, 1],
 [0, 4, 17, 41, 38],
 [0, 1, 3, 16, 67, 12, 1],
 [0, 4, 13, 31, 43, 9],
 [0, 2, 12, 10, 42, 34],
 [0, 2, 5, 24, 43, 25, 1]
],
heli = [
 [0, 0, 7, 27, 61, 5],
 [0, 1, 4, 74, 20, 1],
 [0, 0, 8, 18, 64, 10],
 [0, 1, 5, 43, 46, 5],
 [0, 0, 3, 29, 43, 22, 3],
 [0, 1, 10, 36, 35, 17, 1],
 [0, 2, 7, 25, 58, 8],
 [0, 0, 8, 25, 64, 3],
 [0, 0, 21, 13, 22, 42, 2]
],
glob = [
 [0, 0, 4, 41, 55],
 [0, 4, 43, 42, 9, 2],
 [0, 1, 3, 60, 31, 4, 1],
 [0, 0, 3, 43, 45, 9],
 [1, 4, 16, 51, 28],
 [0, 5, 20, 55, 20],
 [0, 1, 10, 34, 51, 4],
 [0, 5, 24, 50, 20, 1],
 [0, 0, 29, 51, 19, 1]
])

P = [18, 42, 78, 114, 150, 222, 258, 330, 438]
# equi-size to max
pmax = 0
m = 1
for k, v in data.items():
    for u in v:
        m = max(m, len(u))
        pmax = max(pmax, max(u))
print(m)
for k, v in data.items():
    for u in v:
        u.extend([0]*(m-len(u)))

# Share both X and Y axes with all subplots
fig, ax = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(4,3))
fig.subplots_adjust(wspace=0.1)

y, x = np.mgrid[ slice(0,len(P)+1), slice(0.5, m+1) ]
#print(y[:,0]) # 0, 1, 2, ..., 8
#print(x[0]) # 1, 2, 3, ... 7
for i in range(len(P)):
    y[i] = P[i]
y[-1] = P[-1]+100

for i, (k, v) in enumerate( data.items() ):
    z = np.array(v)
    ax[i].set_title(k)
    #ax[i].imshow(x, origin='lowerleft', interpolation='nearest')
    im = ax[i].pcolormesh(x, y, z, cmap='bone_r', vmin=0, vmax=pmax)
    #for u in v:
    #    plt.plot(u)
    ax[i].set_xticks([2,4,6], minor=False)
    ax[i].set_xticks([1,3,5,7], minor=True)

ax[0].set_ylabel(r'$P$')
ax[1].set_xlabel(r'$K$')

fig.colorbar(im, ax=ax[i])
plt.tight_layout()
plt.savefig('prob_K.pdf')
plt.show()
