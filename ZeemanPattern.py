import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 6)
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sin, cos
from copy import deepcopy
from mutils2 import *
import time
from milne import *
plt.switch_backend('TKagg')

# PARAMETROS:
nlinea = 1                      # Numero linea en fichero
param = paramLine(nlinea)

[dpi, dsr, dsb], [spi, ssr, ssb], [ju, lu, su, jl, ll, sl, elem, l0, gu, gl, gef] = param

plt.clf()
for i in range(len(dpi)):
    plt.plot([dpi[i],dpi[i]],[0,spi[i]],'g-')
for i in range(len(dsr)):
    plt.plot([dsr[i],dsr[i]],[0,ssr[i]],'r-')
for i in range(len(dsb)):
    plt.plot([dsb[i],dsb[i]],[0,ssb[i]],'b-')

plt.axhline(0,color='k',ls='dashed')

xmaxi = np.max(dsr)
ymaxi = np.max(spi)
print(xmaxi)
# xmaxi = np.max([spi, ssr, ssb])

plt.xlim(-1.1*xmaxi,+1.1*xmaxi)
plt.ylim(-0.1,1.1*ymaxi)
plt.show()
# plt.savefig('eso.pdf')