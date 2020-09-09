#%% Load Packages
import numpy as np
#%% Activation and Derivatives
def sig(a):
    return 1/(1+np.exp(-a))

def dsig(a):
    return sig(a)*(1-sig(a))

def relu(a):
    return max(0,a)

def drelu(a):
    if a >= 0:
        return 1
    else:
        return 0

def tanh(a):
    return (np.exp(2*a)-1)/(np.exp(2*a)+1)

def dtanh(a):
    return 1-(tanh(a))**2
#%% Loss Function
def CE(in_, out_):
    if in_ == 1:
        return -in_*np.log(out_)
#    elif (in_ == 0 & out_ == 0):
#        return 0
    else:
        return -(1-in_)*np.log(1-out_)

def L2(in_, out_):
    return 0.5*(in_-out_)**2

#%% Forward Propagation
def SGD(X,Y,W,B,LR):
    yout = np.zeros(x.shape[0])
    for i in np.arange(x.shape[0]):
        H1 = X[i,0]*W[0] + X[i,1]*W[1] + B[0] 
        Z1 = tanh(H1)    
        H2 = X[i,0]*W[2] + X[i,1]*W[3] + B[1]
        Z2 = tanh(H2)  
        H3 = Z1*W[4] + Z2*W[5] + B[2]
        Y_hat = tanh(H3)
        yout[i] = Y_hat 
        
        dLdW11 = X[i,0]*dtanh(H1)*W[4]*dtanh(H3)*(Y_hat-Y[i])    
        dLdW12 = X[i,0]*dtanh(H2)*W[5]*dtanh(H3)*(Y_hat-Y[i])
        
        dLdW21 = X[i,1]*dtanh(H1)*W[4]*dtanh(H3)*(Y_hat-Y[i])    
        dLdW22 = X[i,1]*dtanh(H2)*W[5]*dtanh(H3)*(Y_hat-Y[i])    

        dLdW31 = Z1*dtanh(H3)*(Y_hat-Y[i])    
        dLdW32 = Z2*dtanh(H3)*(Y_hat-Y[i])
    
        dLdB1 = dtanh(H1)*W[4]*dtanh(H3)*(Y_hat-Y[i])
        dLdB2 = dtanh(H2)*W[5]*dtanh(H3)*(Y_hat-Y[i])
        dLdB3 = dtanh(H3)*(Y_hat-Y[i])
    
        W[0] = W[0] - dLdW11
        W[2] = W[2] - dLdW12
        W[1] = W[1] - dLdW21
        W[3] = W[3] - dLdW22
        W[4] = W[4] - dLdW31
        W[5] = W[5] - dLdW32
    
        B[0] = B[0] - dLdB1
        B[1] = B[1] - dLdB2
        B[2] = B[2] - dLdB3
    
    return W,B,yout
    
def rinit():
    np.random.seed(seed=0)
    return np.random.randn(6), np.random.randn(3)

def RUN(X,Y,W,B,LR,E):
    for i in np.arange(E):
        W,B,Y_hat = SGD(X,Y,W,B,LR)
        print(W)
        print(B)
        print(Y_hat)
    return W,B,Y_hat
#%%
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])

w, b = rinit()

learning_rate = 0.1 
epochs = 3
t1, t2, yhat = RUN(x,y, w, b, learning_rate, epochs)

#print(t1)
#print(t2)
#%%














