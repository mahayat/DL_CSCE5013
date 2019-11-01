#%%
import numpy as np
#%%
class Conv1D:
    
    def __init__(self, in_channel, out_channel, kernal_size, stride):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernal_size  = kernal_size
        self.stride = stride
        
        self.k = 1/(self.in_channel*self.kernal_size)
        self.W = np.random.uniform(-np.sqrt(self.k), np.sqrt(self.k), (self.out_channel, self.in_channel, self.kernal_size))
        self.b = np.random.uniform(-np.sqrt(self.k), np.sqrt(self.k), self.out_channel)
           

    def __call__(self, data):
        self.data = data
        self.samples, _, self.in_width = np.shape(self.data)           
        self.out_width = ((self.in_width-self.kernal_size)//self.stride)+1
        self.fprop = np.zeros((self.samples, self.out_channel, self.out_width))       
        
        for s in range(self.samples):
            for c in range(self.out_channel):
                for i in range(self.out_width):
                    self.fprop[s,c,i] = np.sum(self.data[s,:,(i*self.stride):((i*self.stride)+self.kernal_size)]*self.W[c]) + self.b[c]
        return self.fprop

   
    def backward(self, delta):        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.dX = np.zeros(self.data.shape)
        
        
        for c in range(self.out_channel):
            for n in range(self.samples):
                for i in range(self.out_width):
                    self.db[c] += delta[n,c,i]
        
        for c in range(self.out_channel):
                for j in range(self.in_channel):
                    for k in range(self.kernal_size):
                        for n in range(self.samples):
                            for i in range(self.out_width):
                                self.dW[c,j,k] += delta[n,c,i]*self.data[n,j,(self.stride*i)+k]
                            
        for n in range(self.samples):
            for j in range(self.in_channel):
                for k in range(self.kernal_size):
                    for c in range(self.out_channel):
                        for i in range(self.out_width):
                            self.dX[n,j,((self.stride*i)+k)] += delta[n,c,i]*self.W[c,j,k]
                                  
        return self.db, self.dW, self.dX

    