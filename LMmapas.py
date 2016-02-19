# ====================================================================================================================================
# ====================================================================================================================================

###############################################################
#  LMEI: Levenberg-Marquardt (with constrain) for a Milne-Eddignton atmosphere Inversion
#
#  CALL:    python LMmapas.py
###############################################################

# ================================================= LIBRARIES
from mutils2 import *
from milne import *
import time
import os
from LMmilne import *
import numpy as np
from math import pi
# from lmfit import minimize, Parameters, fit_report
import matplotlib.pyplot as plt
# from scipy import ndimage

# ================================================= INPUT
nlinea = 3
finalsave = 'finalLMmilne2.npy'
rango = np.arange(32, 130)
wave = np.load('/scratch/carlos/datos_gregor/BINNING/xLambdaBin.npy')
obs = np.load('/scratch/carlos/datos_gregor/BINNING/mapa12B2.npy')
# obs = obs[:, :, 0:20, :]

# ================================================= INPUT2
Chitol = 1e-6
Maxifev = 225
pesoI0 = 1.
pesoQ0 = 2.
pesoU0 = 2.
pesoV0 = 4.
vlos = 0.1
eta0 = 3.                       # Cociente de abs linea-continuo
a = 0.6                         # Parametro de amortiguamiento
ddop = 0.05                         # Anchura Doppler
S_0 = 0.3                         # Ordenada de la funcion fuente
S_1 = 0.6                         # Gradiente de la funcion fuente


# ================================================= LOADING
param = paramLine(nlinea)
l0 = param[2][-4]
wave = wave[rango] - l0
obs = obs[:, :, :, rango]

dx = wave[1] - wave[0]
x = np.arange(wave[0], wave[-1] + dx, dx)
yinver = np.ones((obs.shape[0], obs.shape[2], 11))
npix = obs.shape[0] * obs.shape[2]
ipp = 0.3 * 3.
print('Tiempo estimado: {0:4.2f} s == {1:4.2f} min'.format(
    ipp * npix, ipp * npix / 60.))
time.sleep(5)

# ================================================= INVERSION
time0 = time.time()
for i in range(obs.shape[0]):
    for j in range(obs.shape[2]):
        y2 = obs[i, :, j, :]
        yc = list(y2[0]) + list(y2[1]) + list(y2[2]) + list(y2[3])

        # Modulo Initial conditions:
        iB, igamma, ixi = initialConditions(y2, nlinea, x, param)
        ixi = rad((grad(ixi) + 180.) % 180.)
        igamma = rad((grad(igamma) + 180.) % 180.)

        # Array de valores iniciales
        p = [iB, igamma, ixi, vlos, eta0, a, ddop, S_0, S_1]
        pesoI = 1. * pesoI0
        pesoQ = 1. / max(y2[1]) * pesoQ0
        pesoU = 1. / max(y2[2]) * pesoU0
        pesoV = 1. / max(y2[3]) * pesoV0

        # print('----------------------------------------------------')
        # print('pesos V: {0:2.3f}'.format(pesoV))
        # print('pesos Q, U: {0:2.3f}, {1:2.3f}'.format(pesoQ, pesoU))
        print('quedan: {0:4.1f} s.'.format(
            np.abs(time.time() - time0 - ipp * npix)))

        # Establecemos los pesos
        peso = ones(len(yc))
        peso[0:len(yc) / 4] = pesoI
        peso[len(yc) / 4:3 * len(yc) / 4] = pesoQ
        peso[2 * len(yc) / 4:3 * len(yc) / 4] = pesoU
        peso[3 * len(yc) / 4:] = pesoV

        p0 = Parameters()
        p0.add('B',     value=p[0], min=1.0,  max=2000.)
        p0.add('gamma', value=p[1], min=0.,   max=pi)
        p0.add('xi',    value=p[2], min=0.,   max=pi)
        p0.add('vlos',  value=p[3], min=-5., max=+5.)
        p0.add('eta0',  value=p[4], min=0.,   max=12.)
        p0.add('a',     value=p[5], min=0.,   max=0.8)
        p0.add('ddop',  value=p[6], min=0.0,  max=0.1)
        p0.add('S_0',   value=p[7], min=0.0,  max=1.5)
        p0.add('S_1',   value=p[8], min=0.0,  max=0.7)

        # stokes0 = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
        [ysync, out] = inversionStokes(p0, x, yc, param, Chitol, Maxifev, peso)
        # print('Time: {0:2.4f} s'.format(time.time()-time0))
        # print(fit_report(out, show_correl=False))

        vals = out.params
        yinver[i, j, 0] = vals['B'].value
        yinver[i, j, 1] = grad(vals['gamma'].value)
        yinver[i, j, 2] = grad(vals['xi'].value)
        yinver[i, j, 3] = vals['vlos'].value
        yinver[i, j, 4] = vals['eta0'].value
        yinver[i, j, 5] = vals['a'].value
        yinver[i, j, 6] = vals['ddop'].value
        yinver[i, j, 7] = vals['S_0'].value
        yinver[i, j, 8] = vals['S_1'].value
        yinver[i, j, 9] = out.chisqr
        yinver[i, j, 10] = out.nfev
        # print('nfev: {0}'.format(out.nfev))


print('Time: {0:2.4f} s'.format(time.time() - time0))
print('Time_per_pixel: {0:2.4f} s'.format((time.time() - time0) / npix))
np.save(finalsave, yinver)

# Notificacion 10 s
os.system('notify-send -i face-cool "===> MPyHazel <===" --expire-time=20000')


# PLOT
titulos = ['B', 'thetaB', 'phiB', 'vlos',
           'eta0', 'a', 'ddop', 'S_0', 'S_1', 'chi2']

# plt.figure(1, figsize(18,9))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(yinver[:, :, i], cmap='cubehelix',
               origin='lower', interpolation='none')
    plt.title(titulos[i])
    plt.colorbar()

plt.tight_layout()
plt.show()
