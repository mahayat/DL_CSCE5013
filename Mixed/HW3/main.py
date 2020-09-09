#%% HW3: GRU

import torch
import numpy as np
import torch.nn as nn
import gru as my

#def test():
np.random.seed(11785)
input_dim = 5
hidden_dim = 2
seq_len = 3
timestamp = 0
data = np.random.randn(seq_len, input_dim)

# data = np.random.randint(0, high=10,  size=(seq_len, input_dim)) + np.random.randn(seq_len, input_dim)

#    r1 = gru.Tanh()(data)
#    r2 = nn.Tanh()(torch.Tensor(data))

#%%
g1 = my.GRU_Cell(input_dim, hidden_dim) # your 
g2 = nn.GRUCell(input_dim, hidden_dim, bias=False)
#%%
hidden = np.random.randn(hidden_dim)

g2.weight_ih = nn.Parameter(
    torch.cat([torch.Tensor(g1.Wrx), torch.Tensor(g1.Wzx), torch.Tensor(g1.Wx)]))

g2.weight_hh = nn.Parameter(
    torch.cat([torch.Tensor(g1.Wrh), torch.Tensor(g1.Wzh), torch.Tensor(g1.Wh)]))
#%%
o1 = g1.forward(data[timestamp], hidden)

torch_data = torch.autograd.Variable(torch.Tensor(data[timestamp]).unsqueeze(0), requires_grad=True)
torch_hidden = torch.autograd.Variable(torch.Tensor(hidden).unsqueeze(0), requires_grad=True)

o2 = g2.forward(torch_data, torch_hidden)

#print(o1)
#print(o2)
#%%
delta = np.random.randn(hidden_dim)
o2.backward(torch.Tensor(delta).unsqueeze(0))
delta = delta.reshape(1, -1)
#%%
dx = g1.backward(delta)
dx_t = torch_data.grad
#print(dx)
#print(dx_t)
#%%
#g1_ih_grad = np.concatenate([g1.dWrx, g1.dWzx, g1.dWx], axis=0)
#print(g1_ih_grad)
#g1_hh_grad = np.concatenate([g1.dWrh, g1.dWzh, g1.dWh], axis=0)
#print(g1_hh_grad)
#%% NOK
#g2_dWrx = g2.weight_ih.grad[0:hidden_dim, :]
#print(g1.dWrx)
#print(g2_dWrx)
##%% OK
#g2_dWzx = g2.weight_ih.grad[hidden_dim:2*hidden_dim, :]
#print(g1.dWzx)
#print(g2_dWzx)
##%% OK
#g2_dWx = g2.weight_ih.grad[2*hidden_dim:3*hidden_dim, :]
#print(g1.dWx)
#print(g2_dWx)
#%% NOK
g2_dWrh = g2.weight_hh.grad[0:hidden_dim, :]
print(g1.dWrh)
print(g2_dWrh)
##%% OK
#g2_dWzh = g2.weight_hh.grad[hidden_dim:2*hidden_dim, :]
#print(g1.dWzh)
#print(g2_dWzh)
##%% OK
#g2_dWh = g2.weight_hh.grad[2*hidden_dim:3*hidden_dim, :]
#print(g1.dWh)
#print(g2_dWh)
#%%
#assert(np.allclose(o1, o2.detach().numpy(), atol=1e-2, rtol=1e-2))
##if __name__ == "__main__":
##    test()