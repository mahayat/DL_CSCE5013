import numpy as np

class GRU_cell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = 1/hidden_size
        
        