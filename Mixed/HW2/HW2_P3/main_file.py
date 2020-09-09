#%%
import numpy as np
# Torch Library
import torch
import torch.nn as nn
from torch.autograd import Variable 
# My Library
import layers as my
#%% Create Layers
np.random.seed(10)
net1 = my.Conv1D(8,12,3,2)          # mylib
net2 = torch.nn.Conv1d(8,12,3,2)    # torch
#%% Initialize Layers
x1 = np.random.rand(3,8,20)                             # mylib
x2 = Variable(torch.tensor(x1), requires_grad = True)   # torch
net2.weight = nn.Parameter(torch.tensor(net1.W))        # torch
net2.bias = nn.Parameter(torch.tensor(net1.b))          # torch
#%% Forward Propagation 
y_mylib = net1(x1)                                      # mylib
y_torch = net2(x2)                                      # torch
y_torch_np = y_torch.detach().numpy()                   # torch
#%%
b , c, w = y_mylib.shape
delta = np.random.randn(b,c,w) 
db_mylib, dW_mylib, dX_mylib = net1.backward(delta)
#%%
y_torch.backward(torch.tensor(delta))
dW_torch = net2.weight.grad.detach().numpy()
db_torch = net2.bias.grad.detach().numpy()
dX_torch = x2.grad.detach().numpy()
#%% Compare
def compare(x,y):
    return print(abs(x-y).max())
#%%
compare(y_mylib, y_torch_np)
compare(dX_mylib, dX_torch)
compare(dW_mylib, dW_torch)
compare(db_mylib, db_torch)
    
    
    
    
    
    
    
    