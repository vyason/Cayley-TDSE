#Ankit Kumar - akvyas1995@gmail.com
#subroutines for calculating derivatives with finite-difference approximations
# f =  function, h = grid size, J = dimension of grid
#----------------------------------------------------------------------

import numpy as np

#1st order derivative (3pt central difference)
#-----------------------------------------------------------------------
def diff_3pt(f,h,J):
    d2f=(np.roll(f,-1)-np.roll(f,+1))/(2*h)
    d2f[0]=d2f[J-1]=0
    return d2f

#1st order derivative (5pt central difference)
#-----------------------------------------------------------------------
def diff_5pt(f,h,J):
    df=(-(np.roll(f,-2)-np.roll(f,+2))+8*(np.roll(f,-1)-np.roll(f,+1)))/(12*h)
    df[0]=df[1]=df[J-2]=df[J-1]=0
    return df

#2nd order derivative (3pt central difference)
#-----------------------------------------------------------------------
def diff2_3pt(f,h,J):
    d2f=(np.roll(f,-1)+np.roll(f,+1)-2*f)/h**2
    d2f[0]=d2f[J-1]=0
    return d2f

#2nd order derivative (5pt central difference)
#-----------------------------------------------------------------------
def diff2_5pt(f,h,J):
    d2f=(-(np.roll(f,-2)+np.roll(f,+2))+16*(np.roll(f,-1)+np.roll(f,+1))-30*f)/(12*h**2)
    d2f[0]=d2f[1]=d2f[J-2]=d2f[J-1]=0
    return d2f