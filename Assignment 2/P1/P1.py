import numpy as np
import matplotlib.pyplot as plt

def f(x,w):
    return max(0, x*w)

def dloss(x,w,d):
    if w*x > 0:
        return 2*(f(x,w)-d)*x
    else:
        return 0

w = 1 
dw = 0.5
beta = 0.9
eta = 0.1

x = 0.5
d = 1

dW = dw
W = w

for i in range(100):
    dw = beta*dw - eta*dloss(x,w,d)
    w = w + dw
    dW = np.append(dW,dw)
    W = np.append(W,w)
    
plt.plot(W)
plt.grid()

print(W[1])