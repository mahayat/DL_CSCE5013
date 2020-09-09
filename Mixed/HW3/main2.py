import numpy as np
import torch
import torch.nn as nn

import gru as my
#%%
xdim = 5
hdim = 2
seq_len = 10
np.random.seed(11785)
#%%
data = np.random.randn(seq_len, xdim)
hidden = np.random.randn(hdim)
#%%
g1 = my.GRU_Cell(xdim, hdim)
o1 = g1.forward(data[0], hidden)
print(o1)
#%%
g2 = nn.GRUCell(xdim, hdim, bias=False)
g2.weight_ih = nn.Parameter(
        torch.cat([torch.Tensor(g1.Wrx), torch.Tensor(g1.Wzx), torch.Tensor(g1.Wx)]))

g2.weight_hh = nn.Parameter(
        torch.cat([torch.Tensor(g1.Wrh), torch.Tensor(g1.Wzh), torch.Tensor(g1.Wh)]))

torch_data = torch.autograd.Variable(torch.Tensor(data[0]).unsqueeze(0), requires_grad=True)
torch_hidden = torch.autograd.Variable(torch.Tensor(hidden).unsqueeze(0), requires_grad=True)
o2 = g2.forward(torch_data, torch_hidden)
print(o2)
#%%
assert(np.allclose(o1, o2.detach().numpy(), atol=1e-2, rtol=1e-2))