import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb
phimap = make_colormap([c('white'), c('tomato'), 0.33, c('tomato'), c('deepskyblue'), 0.66, c('deepskyblue'), c('white')])
# phimap = make_colormap([c('white'), c('deepskyblue'), 0.33, c('deepskyblue'), c('tomato'), 0.66, c('tomato'), c('white')])
# phimap = make_colormap([c('white'), c('tomato'), 1/5., c('tomato'), c('darkred'), 2/5., c('darkred'), c('midnightblue'), 3/5., c('midnightblue'), c('deepskyblue'), 4/5., c('deepskyblue'), c('white')])


yinver = np.load('finalLMmilne2.npy')

# PLOT
titulos = ['B', 'thetaB', 'phiB', 'vlos', 'eta0', 'a', 'ddop', 'S_0', 'S_1', 'chi2']

# plt.figure(1, figsize(18,9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(yinver[:, :, i], cmap='cubehelix', origin='lower')
    if i == 2:
        plt.imshow(yinver[:, :, i], cmap=phimap, origin='lower')
    if i == 3:
        norm = MidpointNormalize(midpoint=0)
        plt.imshow(yinver[:, :, i], norm=norm, cmap=plt.cm.seismic, origin='lower')
    plt.title(titulos[i])
    plt.colorbar()

plt.tight_layout()


plt.figure(2)
plt.subplot(2, 1, 1)
plt.imshow(yinver[:, :, 9], cmap='cubehelix', origin='lower', vmax=0.01)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.imshow(yinver[:, :, 10], cmap='cubehelix', origin='lower')
plt.colorbar()

plt.show()
