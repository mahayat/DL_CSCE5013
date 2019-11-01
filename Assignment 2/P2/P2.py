import numpy as np
import matplotlib.pyplot as plt

def f(x,w):
    return 1/(1+np.exp(-w*x))

def loss(x,w,d):
    return -d*np.log(f(x,w))-(1-d)*np.log(1-f(x,w))

def dloss(x,w,d):
    return (f(x,w)-d)*x
#%%
w = 0 
dw = 0.5

beta = 0.9
eta = 0.1

x = -0.5
d = 1

dW = dw
W = w
#%%
for i in range(500):
    dw = beta*dw - eta*dloss(x,w,d)
    w = w + dw
    dW = np.append(dW,dw)
    W = np.append(W,w)
    
plt.plot(W)
plt.grid()