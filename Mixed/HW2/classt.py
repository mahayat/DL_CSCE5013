import numpy as np 

class abul:
    def __init__(self, a):
        self.a = a
    
    def __call__(self, b):
        print(b)
        
#    def dhon(self):
#        print(self.a)
        
kobul = abul(4)
kobul(5)