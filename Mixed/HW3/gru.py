import numpy as np
#import itertools


class Sigmoid:
    # Magic. Do not touch.
    def __init__(self):
        pass

    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res

    def backward(self):
        return self.res * (1-self.res)

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    # Magic. Do not touch.
    def __init__(self):
        pass

    def forward(self, x):
        self.res = np.tanh(x)
        return self.res

    def backward(self):
        return 1 - (self.res**2)

    def __call__(self, x):
        return self.forward(x)


class GRU_Cell:
    def __init__(self, in_dim, hidden_dim):

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.Wrh = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.Wzh = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.Wh = np.random.randn(self.hidden_dim, self.hidden_dim)

        self.Wrx = np.random.randn(self.hidden_dim, self.in_dim)
        self.Wzx = np.random.randn(self.hidden_dim, self.in_dim)
        self.Wx = np.random.randn(self.hidden_dim, self.in_dim)
        
        self.r_act1 = Sigmoid()
        self.r_act2 = Sigmoid()
        self.h_act = Tanh()

    def forward(self, xt, ht):
        # self.ht = h[t-1]
        # self.xt = x[t]
        # self.h = h[t]
        
        self.ht = ht
        self.xt = xt
        
        self.z = self.r_act1(np.matmul(self.Wzh,self.ht) + np.matmul(self.Wzx,self.xt))
        self.r = self.r_act2(np.matmul(self.Wrh,self.ht) + np.matmul(self.Wrx,self.xt))
        self.htilde = self.h_act(np.matmul(self.Wh,(self.r*self.ht)) + np.matmul(self.Wx,self.xt))
        self.h = self.z*self.ht + (1-self.z)*self.htilde
        
        return self.h
        #You may want to store some of the calculated values
#        raise NotImplementedError()

    def backward(self, delta):
        # self.ht = h[t-1]
        # self.xt = x[t]
        # self.h = h[t]
        
#        dg11 = delta 
#        dg12 = delta
        
        dz = -delta*self.htilde
        dhtilde = delta*(1-self.z)
        dz += delta*self.ht
        
        self.dht_1 = delta*self.z
        dg10 = dhtilde*self.h_act.backward()
#        dg8 = dg10
#        dg9 = dg10
#        print(dg10.shape)
        
        self.xt = np.expand_dims(self.xt, axis=0)
        g7 = np.expand_dims(self.r*self.ht, axis=0)
        
        self.dWx = np.matmul(dg10.T, self.xt)

        self.dx = np.matmul(dg10, self.dWx)
        self.dWh = np.matmul(dg10.T, g7)
        
        dg7 = np.matmul(dg10,self.Wh) # (1 x 2) = (1 x 2)*(2 x 2)  
        dr = dg7*self.ht # (1 x 2) = (1 x 2) . (1 x 2)
        self.dht_1 += dg7*dr # 
        
        dg6 = dr*self.r_act2.backward()
        
        self.dWrx = np.matmul(dg6.T,self.xt)
        self.dx += np.matmul(dg6, self.dWrx)

        self.ht = np.expand_dims(self.ht, axis = 0)
        self.dWrh = dg6.T@self.ht
#        print(self.ht)
        self.dht_1 += np.matmul(dg6, self.Wrh)
        
        dg3 = dz*self.r_act1.backward()
        self.dWzx = dg3.T@self.xt
        
        self.dx += dg3@self.Wzx
        
        self.dWzh = dg3.T@self.ht
        
        self.dht_1 += dg3@self.Wzh
        

        return self.dx
        
#        self.dWh = dg10


#        
#        self.dWh 
#        self.dWx 
#        self.dWrh
#        self.dWrx
#        self.dWzh
#        self.dWzx 
#        self.dx 
#        self.dh 
        #And also the derivates of any intermediate values
#        raise NotImplementedError()

