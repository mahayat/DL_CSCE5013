# HW2_P1
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
#def relu_function(input_value, weight):
#    return max(0, weight*input_value)

def d_L2_loss(y, input_value, prev_weight):
    if input_value*prev_weight <= 0:
        return 0
    else:
        return 2*(input_value*prev_weight - y)*input_value

def d_CE_loss(y, input_value, prev_weight):
    if y == 1:
        return (-prev_weight*np.exp(-prev_weight*input_value))/(1+np.exp(-prev_weight*input_value))
    elif y == 0:
        return (prev_weight*np.exp(prev_weight*input_value))/(1+np.exp(prev_weight*input_value))

def updating_parameter(y, input_value, prev_weight, prev_update, loss_type):
    if loss_type == 'L2':
        update = beta*prev_update - eta*d_L2_loss(y, input_value, prev_weight)
    elif loss_type == 'CE':
        update = beta*prev_update - eta*d_CE_loss(y, input_value, prev_weight)
    updated_weight = prev_weight + update
    return updated_weight, update
#%% L2
#new_weight = 1
#new_update = 0.5
#y = 1
#x = 0.5
#%% CE
new_weight = 0
new_update = 0.5
y = 1
x = -0.5
#%%
beta = 0.9
eta = 0.1
W = new_weight
#%% 
for i in np.arange(100):
    new_weight, new_update = updating_parameter(y, x, 
                                                new_weight, new_update, 'CE')
    W = np.append(W,new_weight)
#%%
plt.plot(W, 'r')
plt.grid()
plt.xlabel('Iterations')
plt.ylabel('w')
#%%
    




