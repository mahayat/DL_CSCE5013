# HW2_P1
#%%
def dloss(d, x, w):
    if x*w <= 0:
        return 0
    else:
        return 2*(x*w - d)*x
#%% L2
d = 1
x = 0.5

w = 1
dw = 0.5

beta = 0.9
eta = 0.1
#%% 
w = w + beta*dw - eta*dloss(d, x, w)
#%%
    




