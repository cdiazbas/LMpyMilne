'''
LMEI: Levenberg-Marquardt (with constrain) for a Milne-Eddignton atmosphere Inversion
'''

from mutils2 import *
from milne import *
from numpy import arange, pi, sqrt, array, ones, imag, real, sign, random, diag, load
from lmfit import minimize, Parameters, fit_report

def inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso):

	# Residuals
	def chi2(p,y):
		#vals = p.valuesdict()
		B = p['B'].value
		gamma = p['gamma'].value
		xi = p['xi'].value
		vlos = p['vlos'].value
		eta0 = p['eta0'].value
		a = p['a'].value
		ddop = p['ddop'].value
		S_0 = p['S_0'].value
		S_1 = p['S_1'].value
		
		ysyn= stokesSyn(param,x,B,gamma,xi,vlos,eta0,a,ddop,S_0,S_1)
		ysync = list(ysyn[0])+list(ysyn[1])+list(ysyn[2])+list(ysyn[3])
		ysync = array(ysync)
		return (peso*(ysync-y)**2)

	####################################################################
	#####               Algortimo Levenberg-Marquardt             ######
	#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	n=len(yc); m=len(p0)

	# Levenberg-Marquardt
	out = minimize(chi2, p0, args=(yc,), ftol=Chitol, maxfev=Maxifev)
	#print(fit_report(p0))
	
	## Final fit
	B     = p0['B'].value
	gamma = p0['gamma'].value
	xi    = p0['xi'].value
	vlos  = p0['vlos'].value
	eta0  = p0['eta0'].value
	a     = p0['a'].value
	ddop  = p0['ddop'].value
	S_0   = p0['S_0'].value
	S_1   = p0['S_1'].value
	ysyn  = stokesSyn(param,x,B,gamma,xi,vlos,eta0,a,ddop,S_0,S_1)
	ysync = list(ysyn[0])+list(ysyn[1])+list(ysyn[2])+list(ysyn[3])
	ysync = array(ysync)

	return [ysync,out]
  
if __name__ == "__main__":

	import time
	#time0 = time.time()

	#global param
	global x
	#global peso
	
	
	# PARAMETROS:
	nlinea = 3						# Numero linea en fichero
	x = arange(-0.3,0.3,1e-2)				# Array Longitud de onda
	B = 900. 						# Campo magnetico
	gamma = rad(30.) 					# Inclinacion
	xi = rad(160.) 						# Angulo azimutal
	vlos = 1.1
	eta0 = 3. 						# Cociente de abs linea-continuo
	a = 0.5 						# Parametro de amortiguamiento
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


	# Cargamos los datos:
	# import pyfits as pf
	# hdu = pf.open('/scratch/carlos/datos_gregor/17jun14.006-01ccmnfn.fits')
	# datos = hdu[0].data
	# y2 = datos[74,:,274,24:304]
	y2 = load('stoke3.npy')
	#y2 = datos[87,:,184,24:304]
	#y2 = datos[59,:,430,24:304]
	x = arange(-2.8,2.8,20e-3)
	yc = list(y2[0])+list(y2[1])+list(y2[2])+list(y2[3])
	time0 = time.time()
	
	# Modulo Initial conditions:
	iB, igamma, ixi = initialConditions(y2,nlinea,x,param)
	ixi = rad(( grad(ixi) + 180. ) % 180.)
	igamma = rad(( grad(igamma) + 180. ) % 180.)

	# Array de valores iniciales
	p=[iB,igamma,ixi,vlos,eta0,a,ddop,S_0,S_1]
	
	
	ps = max(y2[0])/max(list(y2[1])+list(y2[2]))
	#print('Peso Q,U sugerido:',ps)
	pesoV = 2./max(y2[3])*0.
	pesoQ = max(pesoQ,ps*2.0)*0.
	pesoU = max(pesoU,ps*2.0)*0.
	
	print('----------------------------------------------------')
	print('pesos V: {0:2.3f}'.format(pesoV))
	print('pesos Q,U: {0:2.3f}, {1:2.3f}'.format(pesoQ,pesoU))
	
	# Establecemos los pesos
	peso = ones(len(yc))
	peso[0:len(yc)/4] = pesoI
	peso[len(yc)/4:3*len(yc)/4] = pesoQ
	peso[2*len(yc)/4:3*len(yc)/4] = pesoU
	peso[3*len(yc)/4:] = pesoV
	

	print('--------------------------------------------------------------------')
	print(' B \t gamma \t xi \t vlos \t eta0 \t a \t ddop \t S_0 \t S_1')
	print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(p[0],grad(p[1]),grad(p[2]),vlos, eta0,a,ddop,S_0,S_1))
	

	from math import pi
	p0 = Parameters()
	#p0.add('B', value=10, vary=False)
	p0.add('B',     value=p[0], min=50.0, max= 2000., vary=False)
	p0.add('gamma', value=p[1], min=0., max = pi, vary=False)
	p0.add('xi',    value=p[2], min=0., max = pi, vary=False)
	p0.add('vlos',  value=p[3]  )
	p0.add('eta0',  value=p[4], min=0., max = 5.0)
	p0.add('a',     value=p[5], min=0., max = 10.0)
	p0.add('ddop',  value=p[6], min=0.0, max = 0.5)
	p0.add('S_0',   value=p[7], min=0.0,  max = 1.5)
	p0.add('S_1',   value=p[8], min=0.0,  max = 1.5)
	

	
	stokes0 = stokesSyn(param,x,B,gamma,xi,vlos,eta0,a,ddop,S_0,S_1)
	
	[ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)
	print('Time: {0:2.4f} s'.format(time.time()-time0))
	
	if grad(out.values['gamma']) < 0.01:
	  	peso[len(yc)/4:3*len(yc)/4] = pesoQ*2.
		peso[2*len(yc)/4:3*len(yc)/4] = pesoU*2.
		[ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)

	  
		
	vals = out.values
	# error = sqrt(diag(out.covar))
	error = ones(len(vals))
	Chi2_min = out.chisqr
	c = out.nfev
	calambda = 0
	n = len(yc)
	m = len(error)
	print('----------------------------------------------------')
	print('Nfev. \t Chi2_min \t tol_Chi lambda   N      M')
	print('----------------------------------------------------')
	print('{0} \t {1:2.2e} \t {2} \t {3} \t {4} \t {5}'.format(c,Chi2_min,Chitol,calambda,n,m))
	print('--------------------------------------------------------------------')
	print(' B\t gamm_g\t xi_g \t vlos \t eta0 \t a \t ddop \t S_0 \t S_1')
	print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(vals['B'],grad(vals['gamma']),grad(vals['xi']),vals['vlos'], vals['eta0'],vals['a'],vals['ddop'],vals['S_0'],vals['S_1']))
	# print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(error[0],grad(error[1]),grad(error[2]),error[3],error[4],error[5],error[6],error[7],error[8]))

	
	

	# 2 CICLO: B,gamma,chi,vlos
	# ============================

	ps = max(y2[0])/max(list(y2[1])+list(y2[2]))
	#print('Peso Q,U sugerido:',ps)
	pesoV = 2./max(y2[3])
	pesoQ = max(pesoQ,ps*3.0)
	pesoU = max(pesoU,ps*3.0)
	
	print('----------------------------------------------------')
	print('pesos V: {0:2.3f}'.format(pesoV))
	print('pesos Q,U: {0:2.3f}, {1:2.3f}'.format(pesoQ,pesoU))
	
	# Establecemos los pesos
	peso = ones(len(yc))
	peso[0:len(yc)/4] = pesoI
	peso[len(yc)/4:3*len(yc)/4] = pesoQ
	peso[2*len(yc)/4:3*len(yc)/4] = pesoU
	peso[3*len(yc)/4:] = pesoV


	p0 = Parameters()
	p0.add('B',     value=p[0], min=50.0, max= 2000., vary=True)
	p0.add('gamma', value=p[1], min=0., max = pi, vary=True)
	p0.add('xi',    value=p[2], min=0., max = pi, vary=True)
	p0.add('vlos',  value=vals['vlos'], vary=False)
	p0.add('eta0',  value=vals['eta0'], min=0.0, max = 10.0, vary=True)
	p0.add('a',     value=vals['a'],    min=0.0, max = 10.0,  vary=False)
	p0.add('ddop',  value=vals['ddop'], min=0.0, max = 0.5,  vary=False)
	p0.add('S_0',   value=vals['S_0'],  min=0.0, max = 1.5,  vary=False)
	p0.add('S_1',   value=vals['S_1'],  min=0.0, max = 1.5,  vary=False)
	

	
	stokes0 = stokesSyn(param,x,B,gamma,xi,vlos,eta0,a,ddop,S_0,S_1)
	
	[ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)
	print('Time: {0:2.4f} s'.format(time.time()-time0))
	
	if grad(out.values['gamma']) < 0.01:
	  	peso[len(yc)/4:3*len(yc)/4] = pesoQ*2.
		peso[2*len(yc)/4:3*len(yc)/4] = pesoU*2.
		[ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)

	  
		
	vals = out.values
	# error = sqrt(diag(out.covar))
	error = ones(len(vals))
	Chi2_min = out.chisqr
	c = out.nfev
	calambda = 0
	n = len(yc)
	m = len(error)
	print('----------------------------------------------------')
	print('Nfev. \t Chi2_min \t tol_Chi lambda   N      M')
	print('----------------------------------------------------')
	print('{0} \t {1:2.2e} \t {2} \t {3} \t {4} \t {5}'.format(c,Chi2_min,Chitol,calambda,n,m))
	print('--------------------------------------------------------------------')
	print(' B\t gamm_g\t xi_g \t vlos \t eta0 \t a \t ddop \t S_0 \t S_1')
	print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(vals['B'],grad(vals['gamma']),grad(vals['xi']),vals['vlos'], vals['eta0'],vals['a'],vals['ddop'],vals['S_0'],vals['S_1']))
	# print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(error[0],grad(error[1]),grad(error[2]),error[3],error[4],error[5],error[6],error[7],error[8]))

	# plot section:
	import matplotlib.pyplot as plt

	stokes = list(split(yc, 4))
	synthetic = list(split(ysync,4))
	for i in range(4):
		plt.subplot(2,2,i+1)
		if i == 0: plt.ylim(0,1.1)
		plt.plot(x,stokes0[i],'g-')
		plt.plot(x,stokes[i],'k-',alpha =0.8)
		plt.plot(x,synthetic[i],'r-')
		#plt.plot([0,0],[min(stokes[i]),max(stokes[i])],'k--')
	plt.tight_layout()
	plt.show()


#----------------------------------------------------
#Nfev. 	 Chi2_min 	 tol_Chi lambda   N      M
#----------------------------------------------------
#246 	 6.21e-04 	 1e-06 	 0 	 1120 	 9
#--------------------------------------------------------------------
# B	 gamm_g	 xi_g 	 vlos 	 eta0 	 a 	 ddop 	 S_0 	 S_1
#1292.1	 38.68	 65.21	 -0.55	 1.51	 5.00	 0.02	 0.3 	 0.5



#----------------------------------------------------
#Nfev. 	 Chi2_min 	 tol_Chi lambda   N      M
#----------------------------------------------------
#280 	 9.95e-06 	 1e-06 	 0 	 1120 	 9
#--------------------------------------------------------------------
# B	 gamm_g	 xi_g 	 vlos 	 eta0 	 a 	 ddop 	 S_0 	 S_1
#1561.1	 41.50	 173.20	 -0.74	 6.00	 4.34	 0.01	 0.4 	 0.4














  
