import numpy as np
import pylab as plt

data = dict(
chomp = [
 [0, 0, 5, 31, 109, 103, 2],
 [0, 0, 6, 68, 116, 56, 4],
 [0, 0, 0, 49, 163, 38],
 [0, 0, 3, 85, 134, 27, 1],
 [0, 0, 0, 10, 142, 96, 2],
 [0, 0, 0, 55, 96, 99],
 [0, 0, 2, 48, 128, 69, 2, 1],
 [0, 0, 3, 25, 187, 34, 1],
 [0, 0, 4, 21, 117, 104, 4],
],
heli = [
 [0, 0, 2, 56, 164, 27, 1],
 [0, 0, 0, 82, 107, 61],
 [0, 0, 1, 60, 161, 27, 1],
 [0, 0, 4, 97, 115, 34],
 [0, 0, 0, 59, 123, 67, 1],
 [0, 1, 6, 33, 83, 121, 6],
 [0, 0, 0, 87, 131, 32],
 [0, 0, 0, 49, 119, 79, 3],
 [0, 0, 6, 74, 115, 55],
],
glob = [
 #[0, 0, 0, 6, 15, 57, 67, 46, 21, 35, 3],
 [0, 0, 0, 0, 0, 0],
 [0, 1, 27, 192, 29, 1],
 [0, 12, 66, 120, 45, 7],
 [1, 7, 79, 106, 46, 11],
 [0, 10, 49, 80, 75, 36],
 [0, 5, 50, 114, 75, 6],
 [0, 12, 57, 105, 74, 2],
 [2, 2, 69, 100, 75, 2],
 [0, 5, 50, 143, 51, 1],
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
