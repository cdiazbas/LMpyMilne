# Author: cdiazbas@iac.es
# Code: Zeeman pattern

# ADDED: velocidad LOS
# ADDED: signo a eta_blue y eta_red
# ADDED: normalization factor in p,b,r eta + rho profiles

from math import pi, sin, cos
from numpy import sqrt, arange
# from zeeman import *
# from fvoigt import fvoigt
from copy import deepcopy
from mutils2 import *

# ============================================
# FUNCTIONS
# ============================================


def grad(x):
    return x * 180. / pi


def rad(x):
    return x * pi / 180.


def stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1):

    # PARAMETROS:
    # param			Parametros de la linea
    # x 			Array Longitud de onda
    # 0 = B			Campo magnetico
    # 1 = gamma		Inclinacion
    # 2 = xi		Angulo azimutal
    # 3 = vlos      Velocidad en la linea de vision [km/s]
    # 4 = eta0		Cociente de abs linea-continuo
    # 5 = a 		Parametro de amortiguamiento
    # 6 = ddop		Anchura Doppler
    # 7 = S_0		Ordenada de la funcion fuente
    # 8 = S_1		Gradiente de la funcion fuente

    # Pi square factor:
    sqrtpi = 1. / sqrt(pi)

    # Magnetic field direction:
    singamma = sin(gamma)
    cosgamma = cos(gamma)
    sin2xi = sin(2 * xi)
    cos2xi = cos(2 * xi)

    # Calculating the Zeeman Pattern
    [[dpi, dsr, dsb], [spi, ssr, ssb], [ju, lu, su, jl,
                                        ll, sl, elem, l0, gu, gl, gef]] = deepcopy(param)

    # Lorentz factor
    lB = 4.67E-13 * l0**2. * B

    # Before in lB (Lorentz units)
    dpi *= lB
    dsr *= lB
    dsb *= lB

    # VLOS: km2A
    cc = 3.0E+5        # veloc luz [km/s]
    vlosA = l0 * vlos / cc

    # ============================================
    # Perfiles de absorcion y dispersion

    # COMPONENTE PI
    # --------------------------------------------
    eta_p = 0.
    rho_p = 0.
    for i in range(0, len(spi)):
        xx = (x - dpi[i] - vlosA) / ddop
        [H, F] = fvoigt(a, xx)
        eta_p = eta_p + H * spi[i] * sqrtpi / ddop
        rho_p = rho_p + 2. * F * spi[i] * sqrtpi / ddop

    # COMPONENTE SIGMA BLUE
    # --------------------------------------------
    eta_b = 0.
    rho_b = 0.
    for i in range(0, len(ssb)):
        xx = (x - dsb[i] - vlosA) / ddop
        [H, F] = fvoigt(a, xx)
        eta_b = eta_b + H * ssb[i] * sqrtpi / ddop
        rho_b = rho_b + 2. * F * ssb[i] * sqrtpi / ddop

    # COMPONENTE SIGMA RED
    # --------------------------------------------
    eta_r = 0.
    rho_r = 0.
    for i in range(0, len(ssr)):
        xx = (x - dsr[i] - vlosA) / ddop
        [H, F] = fvoigt(a, xx)
        eta_r = eta_r + H * ssr[i] * sqrtpi / ddop
        rho_r = rho_r + 2. * F * ssr[i] * sqrtpi / ddop

    # ============================================
    # Elementos la matriz de propagacion

    # 1.- Elemento de absorcion
    eta_I = 1.0 + 0.5 * eta0 * \
        (eta_p * (singamma**2.) + 0.5 * (eta_b + eta_r) * (1. + cosgamma**2.))
    # 2.- Elementos de dicroismo (pol dif en dif direcc)
    eta_Q = 0.5 * eta0 * (eta_p - 0.5 * (eta_b + eta_r)
                          ) * (singamma**2.) * cos2xi
    eta_U = 0.5 * eta0 * (eta_p - 0.5 * (eta_b + eta_r)
                          ) * (singamma**2.) * sin2xi
    eta_V = 0.5 * eta0 * (eta_r - eta_b) * cosgamma
    # 3.- Elementos de dispersion
    rho_Q = 0.5 * eta0 * (rho_p - 0.5 * (rho_b + rho_r)
                          ) * (singamma**2.) * cos2xi
    rho_U = 0.5 * eta0 * (rho_p - 0.5 * (rho_b + rho_r)
                          ) * (singamma**2.) * sin2xi
    rho_V = 0.5 * eta0 * (rho_r - rho_b) * cosgamma

    # ============================================
    # Perfiles de Stokes normalizados al continuo// Sc = S1/S0

    # ScDown = 1.+Sc
    # Det=eta_I**2.*(eta_I**2.-eta_Q**2.-eta_U**2.-eta_V**2.+rho_Q**2.+rho_U**2.+rho_V**2.)-(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V)**2.
    # IDet = 1./Det
    # I=(1.+IDet*eta_I*(eta_I**2.+rho_Q**2.+rho_U**2.+rho_V**2.)*Sc)/ScDown
    # Q=-IDet*(eta_I**2.*eta_Q+eta_I*(eta_V*rho_U-eta_U*rho_V)+rho_Q*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*Sc/ScDown
    # U=-IDet*(eta_I**2.*eta_U+eta_I*(eta_Q*rho_V-eta_V*rho_Q)+rho_U*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*Sc/ScDown
    # V=-IDet*(eta_I**2.*eta_V+eta_I*(eta_U*rho_Q-eta_Q*rho_U)+rho_V*(eta_Q*rho_Q+eta_U*rho_U+eta_V*rho_V))*Sc/ScDown

    Det = eta_I**2. * (eta_I**2. - eta_Q**2. - eta_U**2. - eta_V**2. + rho_Q**2. +
                       rho_U**2. + rho_V**2.) - (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)**2.
    IDet = 1. / Det
    I = S_0 + IDet * eta_I * (eta_I**2. + rho_Q**2. + rho_U**2. + rho_V**2.) * S_1
    Q = -IDet * (eta_I**2. * eta_Q + eta_I * (eta_V * rho_U - eta_U * rho_V) + rho_Q * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)) * S_1
    U = -IDet * (eta_I**2. * eta_U + eta_I * (eta_Q * rho_V - eta_V * rho_Q) + rho_U * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)) * S_1
    V = -IDet * (eta_I**2. * eta_V + eta_I * (eta_U * rho_Q - eta_Q * rho_U) + rho_V * (eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V)) * S_1

    return [I, Q, U, V]
    # return [eta_p,eta_r,eta_b,V]


if __name__ == "__main__":

    # PARAMETROS:
    nlinea = 3						# Numero linea en fichero
    x = arange(-2.8, 2.8, 20e-3)				# Array Longitud de onda
    B = 992.						# Campo magnetico
    gamma = rad(134.)  					# Inclinacion
    xi = rad(145.) 						# Angulo azimutal
    vlos = 0.0                     				 # velocidad km/s
    eta0 = 73. 						# Cociente de abs linea-continuo
    a = 0.2 						# Parametro de amortiguamiento
    ddop = 0.02 						# Anchura Doppler
    # Sc = 4.0						# Cociente Gradiente y Ordenada de la funcion fuente
    S_0 = 0.5							# Ordenada de la funcion fuente
    S_1 = 0.5							# Gradiente de la funcion fuente

    # lambdaStart = 6300.8
    # lambdaStep = 0.015
    # nLambda = 100
    # lambda0 = 6301.5012
    # x = np.arange(lambdaStart,lambdaStart+lambdaStep*nLambda, lambdaStep)-lambda0

    # # VLOS: km2A
    # cc = 3.0E+5
    # l0 = 6301.5012
    # ddop = l0*2./cc

    # class paramlib(object):
    param = paramLine(nlinea)

    # tt = paramlib()
    # print(tt.param)

    stokes = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)

    import matplotlib.pyplot as plt
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.ylim(0, 1.1)
        plt.plot(x, stokes[i])
        plt.plot([0, 0], [min(stokes[i]), max(stokes[i])], 'k--')
        # if i != 0: plt.ylim(-0.4,0.4)
    # plt.tight_layout()
    plt.show()
    # plt.savefig('stokes.pdf')

    # np.save('stokes.npy',stokes)
