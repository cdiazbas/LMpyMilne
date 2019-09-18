'''
LMEI: Levenberg-Marquardt (with constrain) for a Milne-Eddignton atmosphere Inversion
'''

from mutils2 import *
from milne import *
from numpy import arange, pi, array, ones, load
from lmfit import minimize, Parameters, fit_report
# from math import pi


def inversionStokes(p0, x, yc, param, Chitol, Maxifev, peso):

    # Residuals
    def chi2(p, y):
        # vals = p.valuesdict()
        B = p['B'].value
        gamma = p['gamma'].value
        xi = p['xi'].value
        vlos = p['vlos'].value
        eta0 = p['eta0'].value
        a = p['a'].value
        ddop = p['ddop'].value
        S_0 = p['S_0'].value
        S_1 = p['S_1'].value

        ysyn = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
        ysync = list(ysyn[0]) + list(ysyn[1]) + list(ysyn[2]) + list(ysyn[3])
        ysync = array(ysync)
        return (peso * (ysync - y)**2)

    # Algortimo Levenberg-Marquardt

    # n = len(yc)
    # m = len(p0)

    # out = minimize(chi2, p0, args=(yc,), method='leastqr', ftol=Chitol, maxfev=Maxifev)
    out = minimize(chi2, p0, args=(yc,), method='leastsq')#, maxfev=Maxifev)

    # method='leastqr'
    # method='nelder'
    # print(fit_report(p0))

    p0 = out.params
    # Final fit
    B = p0['B'].value
    gamma = p0['gamma'].value
    xi = p0['xi'].value
    vlos = p0['vlos'].value
    eta0 = p0['eta0'].value
    a = p0['a'].value
    ddop = p0['ddop'].value
    S_0 = p0['S_0'].value
    S_1 = p0['S_1'].value
    ysyn = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)
    ysync = list(ysyn[0]) + list(ysyn[1]) + list(ysyn[2]) + list(ysyn[3])
    ysync = array(ysync)

    return [ysync, out]

if __name__ == "__main__":

    import time
    # time0 = time.time()

    global x
    # global peso

    # PARAMETROS:
    nlinea = 3                      # Numero linea en fichero
    x = arange(-0.3, 0.3, 1e-2)               # Array Longitud de onda
    B = 900.                        # Campo magnetico
    gamma = rad(30.)                    # Inclinacion
    xi = rad(160.)                      # Angulo azimutal
    vlos = 1.1
    eta0 = 3.                       # Cociente de abs linea-continuo
    a = 0.2                         # Parametro de amortiguamiento
    ddop = 0.05                         # Anchura Doppler
    S_0 = 0.3                         # Ordenada de la funcion fuente
    S_1 = 0.6                         # Gradiente de la funcion fuente
    Chitol = 1e-6
    Maxifev = 280
    pesoI = 1.
    pesoQ = 4.
    pesoU = 4.
    pesoV = 2.
    param = paramLine(nlinea)

    # Array de valores iniciales
    p = [B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1]

    # Cargamos los datos:
    # import pyfits as pf
    # hdu = pf.open('/scratch/carlos/datos_gregor/17jun14.006-01ccmnfn.fits')
    # datos = hdu[0].data
    # y2 = datos[74,:,274,24:304]
    y2 = load('Profiles/stoke3.npy')
    # y2 = datos[87,:,184,24:304]
    # y2 = datos[59,:,430,24:304]
    x = arange(-2.8, 2.8, 20e-3)
    yc = list(y2[0]) + list(y2[1]) + list(y2[2]) + list(y2[3])
    time0 = time.time()

    # Modulo Initial conditions:
    iB, igamma, ixi = initialConditions(y2, nlinea, x, param)
    ixi = rad((grad(ixi) + 180.) % 180.)
    igamma = rad((grad(igamma) + 180.) % 180.)

    # Array de valores iniciales
    p = [iB, igamma, ixi, vlos, eta0, a, ddop, S_0, S_1]

    ps = max(y2[0]) / max(list(y2[1]) + list(y2[2]))
    # print('Peso Q,U sugerido:',ps)
    pesoV = 1. / max(y2[3])
    pesoQ = 1. / max(y2[1])
    pesoU = 1. / max(y2[2])

    print('----------------------------------------------------')
    print('pesos V: {0:2.3f}'.format(pesoV))
    print('pesos Q,U: {0:2.3f}, {1:2.3f}'.format(pesoQ, pesoU))

    # Establecemos los pesos
    peso = ones(len(yc))
    peso[0:len(yc) / 4] = pesoI
    peso[len(yc) / 4:3 * len(yc) / 4] = pesoQ
    peso[2 * len(yc) / 4:3 * len(yc) / 4] = pesoU
    peso[3 * len(yc) / 4:] = pesoV

    print('--------------------------------------------------------------------')

    p0 = Parameters()
    # p0.add('B', value=10, vary=False)
    p0.add('B',     value=p[0], min=50.0, max=2000.)
    p0.add('gamma', value=p[1], min=0., max=pi)
    p0.add('xi',    value=p[2], min=0., max=pi)
    p0.add('vlos',  value=p[3], min=-5., max=+5.)
    p0.add('eta0',  value=p[4], min=0., max=12.)
    p0.add('a',     value=p[5], min=0., max=0.8)
    p0.add('ddop',  value=p[6], min=0.0, max=0.1)
    p0.add('S_0',   value=p[7], min=0.0,  max=1.5)
    p0.add('S_1',   value=p[8], min=0.0,  max=0.7)

    stokes0 = stokesSyn(param, x, B, gamma, xi, vlos, eta0, a, ddop, S_0, S_1)

    [ysync, out] = inversionStokes(p0, x, yc, param, Chitol, Maxifev, peso)
    print('Time: {0:2.4f} s'.format(time.time() - time0))

    if grad(out.params['gamma'].value) < 0.01:
        peso[len(yc) / 4:3 * len(yc) / 4] = pesoQ * 2.
        peso[2 * len(yc) / 4:3 * len(yc) / 4] = pesoU * 2.
        [ysync, out] = inversionStokes(p0, x, yc, param, Chitol, Maxifev, peso)

    print(fit_report(out, show_correl=False))

    # plot section:
    import matplotlib.pyplot as plt

    stokes = list(split(yc, 4))
    synthetic = list(split(ysync, 4))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.ylim(0, 1.1)
        plt.plot(x, stokes0[i], 'g-')
        plt.plot(x, stokes[i], 'k-', alpha=0.8)
        plt.plot(x, synthetic[i], 'r-')
        # plt.plot([0,0],[min(stokes[i]),max(stokes[i])],'k--')
    # plt.tight_layout()
    plt.show()
    # plt.savefig('prueba2.pdf')
