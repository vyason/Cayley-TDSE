import numpy as np
from math import pi
from scipy.integrate import simps

from calc_diff import *

import os
os.system('rm -rf tridiag pentadiag')


dx = dt = 0.005
plot_steps = 200

# defining the system
#-----------------------------------------------------------------------
hb = 1                               
m =	1                                
hb2m = hb**2/(2*m)                   

x0 = -10                            
p0 = 0                              
sig = 2


xmin,xmax = -25,+25
x = np.arange(xmin,xmax+dx,dx)       
J = len(x)                           

# defining the potential V (for example, HO potential)
#-----------------------------------------------------------------------
w = 0.1
V = 1/2*m*w**2*x**2

w0 = hb/(2*m*sig**2)
tmax = 2*pi/w + 2*plot_steps*dt
for disc_type in ['tridiag','pentadiag']:
    print(f'\n{disc_type}\tdx=dt={dx}')



    psi = np.exp( -((x-x0)/(2*sig))**2 + 1j*p0*(x-x0) )/np.sqrt( sig*np.sqrt(2*pi) )

    engy = ( p0**2+hb**2/(4*sig**2) )/(2*m) + simps(V*np.abs(psi)**2,dx=dx)


    if disc_type == 'tridiag':
        from discretise import lhs_lumatrix_tridiag,zeta_tridiag
        lhs_lumatrix = locals()['lhs_lumatrix_tridiag']
        zeta = locals()['zeta_tridiag']
    elif disc_type == 'pentadiag':
        from discretise import lhs_lumatrix_pentadiag,zeta_pentadiag
        lhs_lumatrix = locals()['lhs_lumatrix_pentadiag']
        zeta = locals()['zeta_pentadiag']
    else:
        raise Exception('Specify disc_type type as tridiag/pentadiag')
    lhs_lu = lhs_lumatrix(J,dx,dt,V,hb,hb2m)	#LU decomposition for the LHS matrix

    # solving the TDSE
    #-----------------------------------------------------------------------
    t = 0
    table = []
    while t < tmax:
        w0t = w0*t
        wt = w*t

        rho = np.abs(psi)**2
        dpsi = diff_5pt(psi,dx,J)
    
        exp_x = simps(x*rho,dx=dx)
        exp_xsq = simps(x**2*rho,dx=dx)
        Dxsq = exp_xsq - exp_x**2

        exp_p = (-1j*hb*simps(np.conj(psi)*dpsi,dx=dx)).real
        exp_psq = 2*m*( engy - simps(V*rho,dx=dx) )

        Dpsq = exp_psq - exp_p**2

        unct = np.sqrt(Dxsq*Dpsq)
        unct_f = hb/2*np.sqrt( np.cos(wt)**4 + np.sin(wt)**4 + 1/4*((w0/w)**2+(w/w0)**2)*np.sin(2*wt)**2 )

        line = f'{wt:f}\t{1-unct/unct_f:e}'
        table.append(line)
        print(line)

        for j in range(plot_steps):
            psi = lhs_lu.solve(zeta(J,psi,dx,dt,V,hb,hb2m))   
        t = t + plot_steps*dt

    np.savetxt(disc_type,table,fmt="%s")


tri = np.genfromtxt('tridiag')
penta = np.genfromtxt('pentadiag')

wt = tri[:,0]
col3 = tri[:,1]
col5 = penta[:,1]

filename = f'errors_cayley_HO_dxdt{dx:.3f}.dat'
filename = 'x'

np.savetxt(filename,np.stack((wt,col5,col3),axis=1),header='w*t (2*pi)\trel err in unct (5)\trel err in unct (3)',fmt="%e")



