'''
LMEI: Levenberg-Marquardt (with constrain) for a Milne-Eddignton atmosphere Inversion
'''

from mutils2 import *
from milne import *
from numpy import arange, pi, sqrt, array, ones, imag, real, sign, random, diag, load
from lmfit import minimize, Parameters, fit_report

def inversionStokes(p0, x, yc, param, Chitol, Maxifev, peso):

    # Residuals
    def chi2(p, y):
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
    out = minimize(chi2, p0, args=(yc,), method='leastqr')#, ftol=Chitol, maxfev=Maxifev)
    # method='leastqr'
    # method='nelder'
    #print(fit_report(p0))
    
    p0 = out.params
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

    global x
    #global peso

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
    pesoV = 1./max(y2[3])
    pesoQ = 1./max(y2[1])
    pesoU = 1./max(y2[2])
    
    print('----------------------------------------------------')
    print('pesos V: {0:2.3f}'.format(pesoV))
    print('pesos Q,U: {0:2.3f}, {1:2.3f}'.format(pesoQ, pesoU))

    # Establecemos los pesos
    peso = ones(len(yc))
    peso[0:len(yc)/4] = pesoI
    peso[len(yc)/4:3*len(yc)/4] = pesoQ
    peso[2*len(yc)/4:3*len(yc)/4] = pesoU
    peso[3*len(yc)/4:] = pesoV


    print('--------------------------------------------------------------------')
    # print(' B \t gamma \t xi \t vlos \t eta0 \t a \t ddop \t S_0 \t S_1')
    # print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(p[0],grad(p[1]),grad(p[2]),vlos, eta0,a,ddop,S_0,S_1))


    from math import pi
    p0 = Parameters()
    #p0.add('B', value=10, vary=False)
    p0.add('B',     value=p[0], min=50.0, max= 2000.)
    p0.add('gamma', value=p[1], min=0., max = pi)
    p0.add('xi',    value=p[2], min=0., max = pi)
    p0.add('vlos',  value=p[3], min=-20., max =+20.)
    p0.add('eta0',  value=p[4], min=0., max = 6.)
    p0.add('a',     value=p[5], min=0., max = 5.0)
    p0.add('ddop',  value=p[6], min=0.0, max = 0.5)
    p0.add('S_0',   value=p[7], min=0.0,  max = 1.5)
    p0.add('S_1',   value=p[8], min=0.0,  max = 1.5)
    

    
    stokes0 = stokesSyn(param,x,B,gamma,xi,vlos,eta0,a,ddop,S_0,S_1)
    
    [ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)
    print('Time: {0:2.4f} s'.format(time.time()-time0))
    
    if grad(out.params['gamma'].value) < 0.01:
        peso[len(yc)/4:3*len(yc)/4] = pesoQ*2.
        peso[2*len(yc)/4:3*len(yc)/4] = pesoU*2.
        [ysync, out] = inversionStokes(p0,x,yc,param,Chitol,Maxifev,peso)
        
    # vals = out.params
    # # error = sqrt(diag(out.covar))
    # error = ones(len(vals))
    # Chi2_min = out.chisqr
    # c = out.nfev
    # calambda = 0
    # n = len(yc)
    # m = len(error)
    # print('----------------------------------------------------')
    # print('Nfev. \t Chi2_min \t tol_Chi lambda   N      M')
    # print('----------------------------------------------------')
    # print('{0} \t {1:2.2e} \t {2} \t {3} \t {4} \t {5}'.format(c,Chi2_min,Chitol,calambda,n,m))
    # print('--------------------------------------------------------------------')
    # print(' B\t gamm_g\t xi_g \t vlos \t eta0 \t a \t ddop \t S_0 \t S_1')
    # print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(vals['B'],grad(vals['gamma']),grad(vals['xi']),vals['vlos'], vals['eta0'],vals['a'],vals['ddop'],vals['S_0'],vals['S_1']))
    # print('{0:3.1f}\t {1:3.2f}\t {2:3.2f}\t {3:1.2f}\t {4:3.2f}\t {5:3.2f}\t {6:3.2f}\t {7:3.1f} \t {8:3.1f}'.format(error[0],grad(error[1]),grad(error[2]),error[3],error[4],error[5],error[6],error[7],error[8]))
    print(fit_report(out, show_correl=False))


    # plot section:
    import matplotlib.pyplot as plt

    stokes = list(split(yc, 4))
    synthetic = list(split(ysync, 4))
    for i in range(4):
        plt.subplot(2,2,i+1)
        if i == 0: plt.ylim(0,1.1)
        plt.plot(x,stokes0[i],'g-')
        plt.plot(x,stokes[i],'k-',alpha =0.8)
        plt.plot(x,synthetic[i],'r-')
        #plt.plot([0,0],[min(stokes[i]),max(stokes[i])],'k--')
    plt.tight_layout()
    plt.show()








    # ----------------------------------------------------
    # Element = SI1
    # lambda0 = 10827.089
    # ju=2.0, lu=1, su=1
    # jl=2.0, ll=1, sl=1
    # g_u  = 1.5
    # g_l  = 1.5
    # geff = 1.5
    # ----------------------------------------------------
    #  B     gamma   xi      eta0    a   ddop    S_0     S_1
    # 850.0  160.00  160.00  100.0   0.80    0.20    0.30    0.5
    # 14.1287234403
    # pesos Q,U: 186.499, 186.499
    # Time: 0.5481 s
    # ----------------------------------------------------
    # Nfev.      Chi2_min    tol_Chi lambda   N      M
    # ----------------------------------------------------
    # 240    9.73e-04    1e-06   0   1120    8
    # ----------------------------------------------------
    #  B     gamm_g  xi_g    eta0    a   ddop    S_0     S_1
    # 753.4  138.11  174.56  58.8    1.00    0.06    0.38    0.6
    # 10.7   0.60    0.89    4.3     0.08    0.00    0.01    0.0


    #----------------------------------------------------
    #Element = SI1
    #lambda0 = 10827.089
    #ju=2.0, lu=1, su=1
    #jl=2.0, ll=1, sl=1
    #g_u  = 1.50
    #g_l  = 1.50
    #geff = 1.50
    #----------------------------------------------------
    #pesos V: 28.257
    #pesos Q,U: 183.142, 183.142
    #--------------------------------------------------------------------
    #B       gamma   xi      vlos    eta0    a       ddop    S_0     S_1
    #922.1    50.63   144.92  2.00    10.00   0.60    0.05    0.3     0.5
    #Time: 0.2219 s
    #----------------------------------------------------
    #Nfev.    Chi2_min        tol_Chi lambda   N      M
    #----------------------------------------------------
    #174      5.75e-04        1e-06   0       1120    9
    #--------------------------------------------------------------------
    #B       gamm_g  xi_g    vlos    eta0    a       ddop    S_0     S_1
    #724.4    45.88   175.15  -0.89   2.78    0.93    0.06    0.3     0.6
    #1.0      57.30   57.30   1.00    1.00    1.00    1.00    1.0     1.0




    ## Modulo Initial conditions:
    #iB, igamma, ixi = initialConditions(y2,nlinea,x)
    ## Array de valores iniciales
    #p=[iB,igamma,ixi,eta0,a,ddop,S_0,S_1]
    #p0.clear()
    #p0.add('B',     value=p[0], min=100.0, max= 2000.)
    #p0.add('gamma', value=p[1], min=0., max = pi)
    #p0.add('xi',    value=p[2], min=0., max = pi)
    #p0.add('eta0',  value=p[3], min=0.0, max = 200.)
    #p0.add('a',     value=p[4], min=0.0, max = 1.5)
    #p0.add('ddop',  value=p[5], min=0.0, max = 1.0)
    #p0.add('S_0',   value=p[6], min=0.0)
    #p0.add('S_1',   value=p[7], min=0.0)
