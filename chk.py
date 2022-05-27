import numpy as np

dx = 0.02
tri = np.genfromtxt('tridiag')
penta = np.genfromtxt('pentadiag')

time = tri[:,0]
col3 = tri[:,1]
col5 = penta[:,1]

np.savetxt(f'errors_cayley_FS_dxdt{dx:.3f}.dat',np.stack((time,col3,col5),axis=1),fmt="%e")
