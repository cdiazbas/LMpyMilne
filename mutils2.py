# Author: cdiazbas@iac.es
# Code: Milne Utilities

from numpy import arange, sqrt, array, ones, imag, real, sign, arctan2, sum
from math import acos
# from zeeman import extrae, g, geff


# Lande factor
def g(j, l, s):
    if j == 0:
        return 0.
    return 3. / 2 + (s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))


# Effective Lande factor
def geff(ju, gu, jl, gl):
    return 0.5 * (gu + gl) + 0.25 * (gl - gu) * (jl * (jl + 1) - ju * (ju + 1))


# Probability of transition S_{i_j}
def sij(ju, mu, jl, ml):
    fPI = []
    fSR = []
    fSB = []
    if ju - jl == 0.:
        if mu - ml == +1.:  # sigma_b
            s = (ju + mu) * (ju - mu + 1.)
            fSB.append(s)
        if mu - ml == 0.:  # pi
            s = 2. * mu**2.
            fPI.append(s)
        if mu - ml == -1.:  # sigma_r
            s = (ju - mu) * (ju + mu + 1.)
            fSR.append(s)
    if ju - jl == +1.:
        if mu - ml == +1.:  # sigma_b
            s = (ju + mu) * (jl + mu)
            fSB.append(s)
        if mu - ml == 0.:  # pi
            s = 2. * (ju**2. - mu**2.)
            fPI.append(s)
        if mu - ml == -1.:  # sigma_r
            s = (ju - mu) * (jl - mu)
            fSR.append(s)
    if ju - jl == -1.:
        if mu - ml == +1.:  # sigma_b
            s = (jl - mu) * (ju - mu + 2.)
            fSB.append(s)
        if mu - ml == 0.:  # pi
            s = 2. * (jl**2. - mu**2.)
            fPI.append(s)
        if mu - ml == -1.:  # sigma_r
            s = (jl + mu) * (ju + mu + 2.)
            fSR.append(s)
    return fPI, fSR, fSB


# Todas las transiciones posibles
def components(ju, jl):
    PI = []
    SR = []
    SB = []
    # M = -J, ..., +J
    mu = arange(-ju, ju + 1, 1)
    ml = arange(-jl, jl + 1, 1)
    # Index for mu & ml
    lonu = arange(0, len(mu), 1)
    lonl = arange(0, len(ml), 1)
    for i in lonu:
        for j in lonl:
            if mu[i] - ml[j] == 0.:
                PI.append((mu[i], ml[j]))
            if mu[i] - ml[j] == -1.:
                SR.append((mu[i], ml[j]))
            if mu[i] - ml[j] == +1.:
                SB.append((mu[i], ml[j]))
    return [PI, SR, SB]


# Split and strength of transition:
def pattern(ju, lu, su, jl, ll, sl):
    dPI = []
    sPI = []
    dSR = []
    sSR = []
    dSB = []
    sSB = []
    gu = g(ju, lu, su)
    gl = g(jl, ll, sl)
    componentsjujl = components(ju, jl)
    for i in componentsjujl[0]:
        delt = gl * i[1] - gu * i[0]
        dPI.append(delt)
        sPI.append(sij(ju, i[0], jl, i[1])[0])
    for i in componentsjujl[1]:
        delt = gl * i[1] - gu * i[0]
        dSR.append(delt)
        sSR.append(sij(ju, i[0], jl, i[1])[1])
    for i in componentsjujl[2]:
        delt = gl * i[1] - gu * i[0]
        dSB.append(delt)
        sSB.append(sij(ju, i[0], jl, i[1])[2])
    sPI = array(sPI).flatten()
    sSR = array(sSR).flatten()
    sSB = array(sSB).flatten()
    dPI = array(dPI).flatten()
    dSR = array(dSR).flatten()
    dSB = array(dSB).flatten()
    return [[dPI, dSR, dSB], [sPI, sSR, sSB]]


# Notation:
def notation(ju, lu, su, jl, ll, sl):
    lNotation = ['S', 'P', 'D', 'F']
    stringU = r'$^' + str(2 * int(su) + 1) + \
        lNotation[int(lu)] + '_' + str(int(ju)) + '$'
    stringL = r'$^' + str(2 * int(sl) + 1) + \
        lNotation[int(ll)] + '_' + str(int(jl)) + '$'
    return stringL + r'$\rightarrow$' + stringU


def extrae(fichero, lineaToFind):
    fileLineas = open(fichero, 'r')
    for linea in fileLineas:
        if int(linea.split('=')[0]) == lineaToFind:
            datos = linea.split()

    ju = float(datos[11][:-1])
    jl = float(datos[9][:-1])

    lNotation = ['S', 'P', 'D', 'F']
    lu = lNotation.index(datos[10][1])
    ll = lNotation.index(datos[8][1])

    su = (int(datos[10][0]) - 1) / 2
    sl = (int(datos[8][0]) - 1) / 2

    elem = (''.join(datos[2:4]))
    l0 = float(datos[4])
    return [ju, lu, su, jl, ll, sl, elem, l0, datos]


def paramLine(nlinea):

    # Calculating the Zeeman Pattern
    lineaToFind = nlinea
    ju, lu, su, jl, ll, sl, elem, l0, datos = extrae('LINEAS.txt', lineaToFind)
    [dpi, dsr, dsb], [spi, ssr, ssb] = pattern(ju, lu, su, jl, ll, sl)

    # Some useful information
    print('----------------------------------------------------')
    print('Element = {0}'.format(elem))
    print('lambda0 = {0}'.format(l0))
    print('ju={0}, lu={1}, su={2}'.format(ju, lu, su))
    print('jl={0}, ll={1}, sl={2}'.format(jl, ll, sl))

    # g(j,l,s); # geff(ju,gu,jl,gl)
    gu = g(ju, lu, su)
    gl = g(jl, ll, sl)
    gef = geff(ju, gu, jl, gl)
    print('g_u  = {0:2.2f}'.format(gu))
    print('g_l  = {0:2.2f}'.format(gl))
    print('geff = {0:2.2f}'.format(gef))

    # Normalization:
    spi /= sum(spi)
    ssr /= sum(ssr)
    ssb /= sum(ssb)

    return [[dpi, dsr, dsb], [spi, ssr, ssb], [ju, lu, su, jl, ll, sl, elem, l0, gu, gl, gef]]


def fvoigt(damp, vv):

    A = [122.607931777104326, 214.382388694706425, 181.928533092181549,
         93.155580458138441, 30.180142196210589, 5.912626209773153,
         0.564189583562615]

    B = [122.60793177387535, 352.730625110963558, 457.334478783897737,
         348.703917719495792, 170.354001821091472, 53.992906912940207,
         10.479857114260399, 1.]

    z = array(damp * ones(len(vv)) + -abs(vv) * 1j)

    Z = ((((((A[6] * z + A[5]) * z + A[4]) * z + A[3]) * z + A[2]) * z + A[1]) * z + A[0]) /\
        (((((((z + B[6]) * z + B[5]) * z + B[4]) * z + B[3]) * z + B[2]) * z + B[1]) * z + B[0])

    h = real(Z)
    f = sign(vv) * imag(Z) * 0.5

    return [h, f]


def split(a, n):
    k, m = len(a) / n, len(a) % n
    return (a[int(i * k + min(i, m)):int((i + 1) * k + min(i + 1, m))] for i in range(n))


def initialConditions(stokes, lineaToFind, x, param):

    # Under weak field:
    I, Q, U, V = stokes
    # I0 = I[:-2]
    I1 = ((I[:-1] - I[1:]) / (x[:-1] - x[1:]))
    I2 = (I1[:-1] - I1[1:]) / (x[:-2] - x[2:])
    I1 = I1[:-1]

    Q0 = Q[:-2]
    U0 = U[:-2]
    V0 = V[:-2]

    [ju, lu, su, jl, ll, sl, elem, l0, gu, gl, gg] = param[2]
    CB = 4.67E-13
    CC = CB * gg * l0**2.
    Bv = -1. / (CC) * (sum(V0 * I1) / sum(I1**2.))
    Bh2 = 4. / (CC**2.) * sqrt(sum(Q0 * I2)**2. +
                               sum(U0 * I2)**2.) / sum(I2**2.)
    Bh = sqrt(Bh2)
    Bm = sqrt(Bv**2. + Bh**2.)
    thetaB = acos(Bv / Bm)
    phiB = 0.5 * arctan2(sum(U0), sum(Q0))
    return Bm, thetaB, phiB
