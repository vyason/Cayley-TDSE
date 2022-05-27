import numpy as np
from math import pi
from scipy.integrate import simps

from calc_diff import *

import os
os.system('rm -rf tridiag pentadiag')


dx = dt = 0.005
plot_steps = 20


hb = 1                               
m =	1                                
hb2m = hb**2/(2*m)                   

x0 = -50                            
p0 = 1                              
sig = 2


xmin,xmax = -75,+125
x = np.arange(xmin,xmax+dx,dx)       
J = len(x)                           

# defining the potential V (for example, HO potential)
#-----------------------------------------------------------------------
V = np.zeros(J,int)


w0 = hb/(2*m*sig**2)
tmax = 50 + 2*plot_steps*dt
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
        w0t= w0*t

        rho = np.abs(psi)**2
        psi_st = np.conj(psi)
    
        exp_x = simps(x*rho,dx=dx)
        exp_xsq = simps(x**2*rho,dx=dx)
        Dxsq = exp_xsq - exp_x**2

        exp_p = (-1j*hb*simps(psi_st*diff_5pt(psi,dx,J),dx=dx)).real
        exp_psq = 2*m*( engy - simps(V*rho,dx=dx) )
        Dpsq = exp_psq - exp_p**2

        unct = np.sqrt(Dxsq*Dpsq)
        unct_f = hb/2*np.sqrt(1+(w0*t)**2)

        line = f'{t:f}\t{1-unct/unct_f:e}'
        table.append(line)
        print(line)

        for j in range(plot_steps):
            psi = lhs_lu.solve(zeta(J,psi,dx,dt,V,hb,hb2m))   
        t = t + plot_steps*dt

    np.savetxt(disc_type,table,fmt="%s")


tri = np.genfromtxt('tridiag')
penta = np.genfromtxt('pentadiag')

time = tri[:,0]
col3 = tri[:,1]
col5 = penta[:,1]

filename = f'errors_cayley_FS_dxdt{dx:.3f}.dat'
np.savetxt(filename,np.stack((time,col5,col3),axis=1),header=f'w*t (2*pi)\trel err in unct (5)\trel err in unct (3)\nhb=m={m}\tV=0\tdx=dt={dx}\tx0={x0}\tp0={p0}',fmt="%e")
os.system(f'cp {filename} x')

