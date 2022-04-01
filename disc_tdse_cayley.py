#Ankit Kumar - akvyas1995@gmail.com
#subroutines for discretising TDSE with the Cayley's operator
# J = dimension of grid, dx = grid size, dt = time step
# V = potential, hb = hbar, hb2m = hbar^2/2m, psi = input wave function
#----------------------------------------------------------------------

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu

from calc_diff import *

#LU matrix on the left: tridiagonal case
#----------------------------------------------------------------------
def lhs_lumatrix_tridiag(J,dx,dt,V,hb,hb2m):
    a = 1 + 1j*dt*(2*hb2m/dx**2 + V)/(2*hb)
    b = (-1j*hb2m*dt/(2*hb*dx**2))*np.ones((J),float)
    return splu(spdiags(np.array([b,a,b]),np.array([-1,0,+1]),J,J).tocsc())

#zeta vector on the right: tridiagonal case
#----------------------------------------------------------------------
def zeta_tridiag(J,psi,dx,dt,V,hb,hb2m):
    return psi - 1j*(dt/hb)*(-hb2m*diff2_3pt(psi,dx,J)+V*psi)/2

#LU matrix on the left: pentadiagonal case
#----------------------------------------------------------------------
def lhs_lumatrix_pentadiag(J,dx,dt,V,hb,hb2m):
    a = 1 + 1j*dt*(5*hb2m/(2*dx**2) + V)/(2*hb)
    b = (-2*1j*hb2m*dt/(3*hb*dx**2))*np.ones((J),float)
    c = -b/16
    return splu(spdiags(np.array([c,b,a,b,c]),np.array([-2,-1,0,+1,+2]),J,J).tocsc())

#zeta vector on the right: pentadiagonal case
#----------------------------------------------------------------------
def zeta_pentadiag(J,psi,dx,dt,V,hb,hb2m):
    return psi - 1j*(dt/hb)*(-hb2m*diff2_5pt(psi,dx,J)+V*psi)/2