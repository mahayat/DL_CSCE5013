import numpy as np
n = 100
n1 = 60
nL = [50, 30, 80]
n1L = [30, 20, 50]

n0L = np.subtract(nL,n1L)
n0 = n - np.asarray(n1)
nR = 100 - np.asarray(nL)
n1R = n1-np.asarray(n1L)
n0R = nR-np.asarray(n1R)

G = []
for i in range(3):
    G.append((2*n0*n1)/n - (2*n0L[i]*n1L[i])/nL[i] - (2*n0R[i]*n1R[i])/nR[i]) 
