from types import SimpleNamespace
import numpy as np
import scipy

from Funcs import *


class graduate_model:
    def __init__(self):
        '''Initialize the model'''
        self.par = SimpleNamespace()

        self.setup()
    
    def setup(self):
        '''Defined parameters'''

        par = self.par
        
        par.J = 3
        par.N = 10
        par.K = 10000

        par.F = np.arange(1, par.N+1)
        par.sigma = 2

        par.v = np.array([1, 2, 3])
        par.c = 1