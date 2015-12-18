'''
LMEI: Levenberg-Marquardt (with constrain) for a Milne-Eddignton atmosphere Inversion
'''

from LMEI2 import *
from numpy import ones
import time
from math import pi
time0 = time.time()


# PARAMETROS:
nlinea = 3						# Numero linea en fichero
x = arange(-0.3,0.3,1e-2)				# Array Longitud de onda
B = 900. 						# Campo magnetico [G]
gamma = rad(160.) 					# Inclinacion
xi = rad(160.) 						# Angulo azimutal
vlos = 1.1						# Velocidad vertical LOS [km/s]
eta0 = 3. 						# Cociente de abs linea-continuo
a = 0.2 						# Parametro de amortiguamiento
ddop = 0.05 						# Anchura Doppler
S_0=0.3							# Ordenada de la funcion fuente
S_1=0.6							# Gradiente de la funcion fuente
Chitol = 1e-6
Maxifev = 280
pesoI = 1.
pesoQ = 4.
pesoU = 4.
pesoV = 2.
param = paramLine(nlinea)

# Array de valores iniciales
p=[B,gamma,xi,vlos,eta0,a,ddop,S_0,S_1]

# Los anadimos:
p0 = Parameters()
p0.add('B',     value=p[0], min=50.0, max= 2000.)
p0.add('gamma', value=p[1], min=0.,   max = pi)
p0.add('xi',    value=p[2], min=0.,   max = pi)
p0.add('vlos',  value=p[3], min=-5.0, max = +5.0)
p0.add('eta0',  value=p[4], min=0.,   max = 6.)
p0.add('a',     value=p[5], min=0.,   max = 2.0)
p0.add('ddop',  value=p[6], min=0.0,  max = 0.5)
p0.add('S_0',   value=p[7], min=0.0,  max = 1.5)
p0.add('S_1',   value=p[8], min=0.0,  max = 1.5)


# Cargamos los datos:
import pyfits as pf
hdu = pf.open('/scratch/carlos/datos_gregor/17jun14.006-01ccmnfn.fits')
datos = hdu[0].data


#yprim = datos[40:80,:,420:460,24:304]
#yprim = datos[50:60,:,430:440,24:304]
yprim = datos[:,:,:,24:304]
yinver = ones((yprim.shape[0],yprim.shape[2],10))
print(yinver.shape[:],yprim.shape[3]*4)
npix = yprim.shape[0]*yprim.shape[2]

maxfevArray = []
nofunciona  = []
pixelArray  = []
cont= 0
for i in range(yprim.shape[0]):
	for j in range(yprim.shape[2]):
	
	  pixelArray.append([i,j])
	  cont += 1
	  
	  y2 = yprim[i,:,j,:]
	  x = arange(-2.8,2.8,20e-3)
	  yc = list(y2[0])+list(y2[1])+list(y2[2])+list(y2[3])
	  
	  # Modulo Initial conditions:
	  iB, igamma, ixi = initialConditions(y2,nlinea,x,param)
	  ixi = rad(( grad(ixi) + 180. ) % 180.)
	  igamma = rad(( grad(igamma) + 180. ) % 180.)
	  
	  # Array de valores iniciales
	  p=[iB,igamma,ixi,vlos,eta0,a,ddop,S_0,S_1]
	  #p0.clear()
	  p0.pop('B')
	  p0.add('B',     value=p[0], min=50.0, max= 2000.)
	  p0.pop('gamma')
	  p0.add('gamma', value=p[1], min=0., max = pi)
	  p0.pop('xi')
	  p0.add('xi',    value=p[2], min=0., max = pi)

	  qq = 0
	  if qq == 0:
	  #try:
		  ps = max(y2[0])/max(list(y2[1])+list(y2[2]))
		  #print('Peso Q,U sugerido:',ps)
		  pesoVn = max(2./max(y2[3]),pesoV)
		  pesoQn = max(pesoQ,ps*2.0)
		  pesoUn = max(pesoU,ps*2.0)
		  print('pesos Q,U,V: {0:2.3f}, {1:2.3f}, {2:2.3f}'.format(pesoQn,pesoUn,pesoVn))
			  
		  # Establecemos los pesos
		  peso = ones(len(yc))
		  peso[0:len(yc)/4] = pesoI
		  peso[len(yc)/4:2*len(yc)/4] = pesoQn
		  peso[2*len(yc)/4:3*len(yc)/4] = pesoUn
		  peso[3*len(yc)/4:] = pesoVn

		  print('... {0:3.2f}%'.format(float(cont)/npix*100.))
		  
		  [ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)
		  #print('Time: {0:2.4f} s'.format(time.time()-time0))
		  
		  if grad(out.values['gamma']) < 0.1:
			peso[len(yc)/4:3*len(yc)/4] = pesoQn*2.
			peso[2*len(yc)/4:3*len(yc)/4] = pesoUn*2.
			[ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)
			  
		  vals = out.values
		  #error = sqrt(diag(out.covar))
		  error = ones(9)
		  Chi2_min = out.chisqr
		  c = out.nfev
		  maxfevArray.append(c)
		  calambda = 0
		  n = len(yc)
		  m = len(error)
		  
		  # Guardamos los datos:
		  yinver[i,j,0] = vals['B']
		  yinver[i,j,1] = grad(vals['gamma'])
		  yinver[i,j,2] = grad(vals['xi'])
		  yinver[i,j,3] = vals['vlos']
		  yinver[i,j,4] = vals['eta0']
		  yinver[i,j,5] = vals['a']
		  yinver[i,j,6] = vals['ddop']
		  yinver[i,j,7] = vals['S_0']
		  yinver[i,j,8] = vals['S_1']
		  yinver[i,j,9] = out.chisqr
		  
		  
		  # para mejorara la inicializacion de a en la prox
		  p0.pop('eta0')
		  p0.pop('a')
		  p0.pop('ddop')
		  p0.pop('S_0')
		  p0.pop('S_1')
		  p0.add('eta0',  value=p[4], min=0.0, max = 6.)
		  p0.add('a',     value=p[5], min=0.0, max = 2.)
		  p0.add('ddop',  value=p[6], min=0.0, max = 0.5)
		  p0.add('S_0',   value=p[7], min=0.0, max = 1.5)
		  p0.add('S_1',   value=p[8], min=0.0, max = 1.5)
	  else:
	  #except:
		  print('error en',i,j)
		  nofunciona.append([i,j])

	  

print('Time: {0:2.4f} s'.format(time.time()-time0))
print('Time_per_pixel: {0:2.4f} s'.format((time.time()-time0)/npix))

print('Errores en:',nofunciona)

#Correcion xi
#yinver[:,:,2] += 360.
#yinver[:,:,2] = yinver[:,:,2]%360

# plot section:
import matplotlib.pyplot as plt

titulos = ['B','gamma','xi','vlos','eta0','a','ddop','S_0','S_1','chi2']

# plt.figure(1, figsize(18,9))
for i in range(9):
	plt.subplot(3,3,i+1)
	plt.imshow(yinver[:,:,i],cmap = 'cubehelix', origin='lower', interpolation='none')
	plt.title(titulos[i])
	plt.colorbar()

plt.tight_layout()
plt.show()


plt.figure(2)
plt.imshow(yinver[:,:,9],cmap = 'cubehelix', origin='lower', interpolation='none')
plt.title(titulos[9])
plt.colorbar()
plt.tight_layout()
#plt.show()

plt.figure(3)
plt.hist(maxfevArray,50)
plt.show()


import numpy as np
np.save('/scratch/carlos/datos_gregor/invertidoLMEI2ALL.npy',yinver)
# plt.savefig('invertidoALL.pdf')
	  
	  
	  
	  
	  
	  
	  

	  
	  
	  
