#Ankit Kumar - akvyas1995@gmail.com
#python code for solving the TDSE usingCayley's operator
#----------------------------------------------------------------------

import numpy as np
from math import pi
from scipy.integrate import simps

from calc_diff import *

#specify the discretisation type as 'tridiag' or 'pentadiag'
#----------------------------------------------------------------------
disc_type = 'pentadiag'

#defining the system
#----------------------------------------------------------------------
hb = 1                                  #Planck's constant, hbar
m =	1                                   #mass of particle, m

xmin,xmax = -25,+25                     #x-limits of simulation box
dx = 0.01                               #grid size, dx 
x = np.arange(xmin,xmax+dx,dx)          #defining the position grid
J = len(x)                              #dimension of position grid

#defining the potential V (for example, HO potential)
#----------------------------------------------------------------------
w = 0.1
V = 1/2*m*w**2*x**2

#defining the initial wave packet (for example, Gaussian)
#----------------------------------------------------------------------
x0 = -10                                #initial position
p0 = 0                                  #initial momentum            
sig = 2                                 #initial position spread

#wave function
psi = np.exp( -((x-x0)/(2*sig))**2 + 1j*p0*(x-x0) )/np.sqrt( sig*np.sqrt(2*pi) )

#time limits for the simulation
#----------------------------------------------------------------------
tmax = 2*pi/w                           #simulation time limit
dt = 0.01                               #time step, dt
plot_steps = 10                         #time steps b/w two successive print statements

#setting up the numerical structure
#----------------------------------------------------------------------
hb2m = hb**2/(2*m)                                          #value of hbar^2/2m
if disc_type == 'tridiag':
    from disc_tdse_cayley import lhs_lumatrix_tridiag,zeta_tridiag
    lhs_lumatrix = locals()['lhs_lumatrix_tridiag']
    zeta = locals()['zeta_tridiag']
elif disc_type == 'pentadiag':
    from disc_tdse_cayley import lhs_lumatrix_pentadiag,zeta_pentadiag
    lhs_lumatrix = locals()['lhs_lumatrix_pentadiag']
    zeta = locals()['zeta_pentadiag']
else:
    raise Exception('Specify disc_type type as tridiag/pentadiag')
lhs_lu = lhs_lumatrix(J,dx,dt,V,hb,hb2m)	#LU decomposition for the LHS matrix

#solving the TDSE
#----------------------------------------------------------------------
t = 0
while t < tmax:
    ex = simps(x*np.abs(psi)**2,dx=dx)                      #<x>
    dpsi = diff_5pt(psi,dx,J)                               #d(psi)/dx
    ep = (-1j*hb*simps(np.conj(psi)*dpsi,dx=dx)).real       #<p>

    print(f'{w*t:f}\t{ex:e}\t{ep:e}')

    for j in range(plot_steps):                             #evolve plot_steps times
        psi = lhs_lu.solve(zeta(J,psi,dx,dt,V,hb,hb2m))   
    t = t + plot_steps*dt