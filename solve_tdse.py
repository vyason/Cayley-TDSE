# python code for solving the TDSE using Cayley's operator
# Ankit Kumar, kumar.ankit.vyas@gmail.com
#-----------------------------------------------------------------------

import numpy as np
from math import pi
from scipy.integrate import simpson
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

# 1st order derivative (regular and improved central difference)
#-----------------------------------------------------------------------
def diff(f,h,J,diff_type="regular"):

    if diff_type=="regular":
        df = (np.roll(f,-1)-np.roll(f,+1))/(2*h)
        df[0] = df[J-1] = 0

    elif diff_type=="improved":
        df = (-(np.roll(f,-2)-np.roll(f,+2))+8*(np.roll(f,-1)-np.roll(f,+1)))/(12*h)
        df[0] = df[1] = df[J-2] = df[J-1] = 0

    return df


# 2nd order derivative (regular and improved central difference)
#-----------------------------------------------------------------------
def diff2(f,h,J,diff_type="regular"):

    if diff_type=="regular":
        d2f = (np.roll(f,-1)+np.roll(f,+1)-2*f)/h**2
        d2f[0] = d2f[J-1] = 0

    elif diff_type=="improved":
        d2f = (-(np.roll(f,-2)+np.roll(f,+2))+16*(np.roll(f,-1)+np.roll(f,+1))-30*f)/(12*h**2)
        d2f[0] = d2f[1] = d2f[J-2] = d2f[J-1] = 0

    return d2f


#calculate the zeta vector that appears on the RHS
#---------------------------------------------------------------
def calc_zeta(J,psi,dx,dt,V,hb,hb2m,sim_type="regular"):
    return psi - 1j*(dt/hb)*(-hb2m*diff2(psi,dx,J,sim_type)+V*psi)/2


# LU matrix on the left: regularonal case
#-----------------------------------------------------------------------
def calc_lumatrix(J,dx,dt,V,hb,hb2m,sim_type="regular"):
    
    if sim_type=="regular":
        a = 1 + 1j*dt*(2*hb2m/dx**2 + V)/(2*hb)
        b = (-1j*hb2m*dt/(2*hb*dx**2))*np.ones((J),float)
        lumatrix = splu(spdiags(np.array([b,a,b]),np.array([-1,0,+1]),J,J).tocsc())

    elif sim_type=="improved":
        a = 1 + 1j*dt*(5*hb2m/(2*dx**2) + V)/(2*hb)
        b = (-2*1j*hb2m*dt/(3*hb*dx**2))*np.ones((J),float)
        c = (2*1j*hb2m*dt/(48*hb*dx**2))*np.ones((J),float)
        lumatrix = splu(spdiags(np.array([c,b,a,b,c]),np.array([-2,-1,0,+1,+2]),J,J).tocsc())

    return lumatrix


# specify the discretisation type as 'regular' or 'improved'
#-----------------------------------------------------------------------


# defining the system
#-----------------------------
sim_type = "improved"
hb = 1                                  #Planck's constant, hbar
m =	1                                   #mass of particle, m
hb2m = hb**2/(2*m)                                          #value of hbar^2/2m

xmin,xmax = -10,+10                     #x-limits of simulation box
dx = 0.01                               #grid size, dx 
x = np.arange(xmin,xmax+dx,dx)          #defining the position grid
J = len(x)                              #dimension of position grid

# potential and initial wave function
#------------------------------------------
V = np.zeros(J,int)
V[0] = V[-1] = 1e10 

psi = np.exp(-x**2 + 1j*x )

tmax = 25                              #simulation time limit
dt   = 0.01                            #time step, dt
plot_steps = 10                         #time steps b/w two successive plot updates


# solving the TDSE
#-----------------------------------------------------------------------
t = 0
lumatrix = calc_lumatrix(J,dx,dt,V,hb,hb2m,sim_type)
while t < tmax:
    dpsi = diff(psi,dx,J,"improved")

    exp_x = simpson(x*np.abs(psi)**2,dx=dx)
    exp_p = (-1j*hb*simpson(np.conj(psi)*dpsi,dx=dx)).real 
    norm  = simpson(np.abs(psi)**2,dx=dx)        

    print(f"{t:f} \t {norm:f}")

    for _ in range(plot_steps):
        psi = lumatrix.solve(calc_zeta(J,psi,dx,dt,V,hb,hb2m,sim_type))   
    t = t + plot_steps*dt

    plt.cla()
    plt.plot(x,np.abs(psi)**2)      # plot x vs |psi|^2
    plt.grid()
    plt.title(f"t = {t:f}")
    plt.pause(0.1)
