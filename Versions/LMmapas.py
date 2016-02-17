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
from LMmilne import *
import numpy as np
from math import pi
# from lmfit import minimize, Parameters, fit_report
import matplotlib.pyplot as plt

# ================================================= INPUT
nlinea = 3
param = paramLine(nlinea)
l0 = param[2][-4]
maxiteDIRECT = 20
maxiteLM = -1  # -1 para DeltaChi2Realtivo = e-7
mode = 'useLast'
rango = np.arange(32, 130)
wave = np.load('/scratch/carlos/datos_gregor/BINNING/xLambdaBin.npy')
wave = wave[rango] - l0
obs = np.load('/scratch/carlos/datos_gregor/BINNING/mapa12B2.npy')
obs = obs[:, :, :, rango]
# obs = obs[:, :, 0:20, :]


# ================================================= INPUT2
y2 = obs[0, :, 0, :]
B = 900.                        # Campo magnetico
gamma = rad(30.)                    # Inclinacion
xi = rad(160.)
vlos = 1.1
eta0 = 3.                       # Cociente de abs linea-continuo
a = 0.5                         # Parametro de amortiguamiento
ddop = 0.05                         # Anchura Doppler
S_0 = 0.3                         # Ordenada de la funcion fuente
S_1 = 0.6                         # Gradiente de la funcion fuente
Chitol = 1e-6
Maxifev = 280
pesoI0 = 1.
pesoQ0 = 2.
pesoU0 = 2.
pesoV0 = 4.

# ================================================= INVERSION
dx = wave[1]-wave[0]
x = np.arange(wave[0], wave[-1]+dx, dx)
yc = list(y2[0])+list(y2[1])+list(y2[2])+list(y2[3])
time0 = time.time()

# Modulo Initial conditions:
iB, igamma, ixi = initialConditions(y2, nlinea, x, param)
ixi = rad((grad(ixi) + 180.) % 180.)
igamma = rad((grad(igamma) + 180.) % 180.)

# Array de valores iniciales
p = [iB, igamma, ixi, vlos, eta0, a, ddop, S_0, S_1]
pesoI = 1.*pesoI0
pesoQ = 1./max(y2[1])*pesoQ0
pesoU = 1./max(y2[2])*pesoU0
pesoV = 1./max(y2[3])*pesoV0

print('----------------------------------------------------')
print('pesos V: {0:2.3f}'.format(pesoV))
print('pesos Q, U: {0:2.3f}, {1:2.3f}'.format(pesoQ, pesoU))

# Establecemos los pesos
peso = ones(len(yc))
peso[0:len(yc)/4] = pesoI
peso[len(yc)/4:3*len(yc)/4] = pesoQ
peso[2*len(yc)/4:3*len(yc)/4] = pesoU
peso[3*len(yc)/4:] = pesoV


# print('--------------------------------------------------------------------')
# print(' B \t gamma \t xi \t vlos \t eta0 \t a \t ddop \t S_0 \t S_1')
# print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(p[0],grad(p[1]),grad(p[2]),vlos, eta0,a,ddop,S_0,S_1))

p0 = Parameters()
p0.add('B',     value=p[0], min=50.0, max=2000.)
p0.add('gamma', value=p[1], min=0.,   max=pi)
p0.add('xi',    value=p[2], min=0.,   max=pi)
p0.add('vlos',  value=p[3], min=-20., max=+20.)
p0.add('eta0',  value=p[4], min=0.,   max=50.)
p0.add('a',     value=p[5], min=0.,   max=0.8)
p0.add('ddop',  value=p[6], min=0.0,  max=0.5)
p0.add('S_0',   value=p[7], min=0.0,  max=1.5)
p0.add('S_1',   value=p[8], min=0.0,  max=1.5)

stokes0 = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
[ysync, out] = inversionStokes(p0, x, yc, param, Chitol, Maxifev, peso)
print('Time: {0:2.4f} s'.format(time.time()-time0))
print(fit_report(out, show_correl=False))

stokes = list(split(yc, 4))
synthetic = list(split(ysync, 4))
for i in range(4):
    plt.subplot(2, 2, i+1)
    if i == 0:
        plt.ylim(0, 1.1)
    plt.plot(x, stokes0[i], 'g-')
    plt.plot(x, stokes[i], 'k-', alpha=0.8)
    plt.plot(x, synthetic[i], 'r-')
    # plt.plot([0,0],[min(stokes[i]),max(stokes[i])],'k--')
    # plt.tight_layout()
plt.show()
