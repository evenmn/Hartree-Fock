import numpy as np
from Matrix_elements import *

class CIS:
    '''Configuration Interction Singles class for atomic structure'''
    
    def __init__(self, Z, basis):
        '''
        Arguments:
        ----------
        Z:      Int.
                Atomic number (proton number)
        '''
        
        self.elem = Integrals(Z,basis)  # Matrix elements/integrals
        self.n = Z                      # Assume neutral atom


    def c_H_c(self):
        '''Reference energy'''
        
        OBT = 0
        for i in range(self.n):
            OBT += self.elem.OBME(i,i)
                
        TBT = 0
        for i in range(self.n):
            for j in range(self.n):
                TBT += 0.5*self.elem.AS(i,j,i,j)
        
        return OBT + TBT
        
        
    def c_H_ia(self, i,a):
        '''Singly excited ket'''
        
        OBT = self.elem.OBME(i,a)
        
        TBT = 0
        for j in range(self.n):
            TBT += self.elem.AS(a,j,i,j)
        
        return OBT + TBT
        
        
    def ia_H_jb(self, i,a,j,b):
        '''Singly excited bra and ket'''
        
        Result = self.elem.AS(a,j,i,b)
        
        if a==b:
            Result -= self.elem.OBME(i,j)
            for k in range(self.n):
                Result -= self.elem.AS(i,k,j,k)
                    
            if i==j:
                for k in range(self.n):
                    Result += self.elem.OBME(k,k)
                    for l in range(self.n):
                        Result += 0.5*self.elem.AS(k,l,k,l)
                                
        if i==j:
            Result += self.elem.OBME(a,b)
            for k in range(self.n):
                Result += self.elem.AS(a,k,b,k)
        
        return Result
