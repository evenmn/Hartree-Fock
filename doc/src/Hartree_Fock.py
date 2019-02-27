import numpy as np
from hydrogenorbitals import *


class HF:
    '''Hartree-Fock class for atomic structure'''
    
    def __init__(self, Z, basis):
        '''
        Arguments:
        ----------
        Z:      Int.
                Atomic number (proton number)
        '''
        
        self.elem = Integrals(Z,basis)  # Matrix elements/integrals
        self.n = Z                      # Number of electrons, assuming neutral atom
        self.N = len(basis)             # Number of states


    def HF_elements(self,i,j,C):
        '''HF matrix elements (i,j)'''
        
        Result = self.elem.OBME(i, j)
        
        for p in range(self.n):
            for c in range(self.N):
                for d in range(self.N):
                    Result += C[p,c]*C[p,d]*self.elem.AS(i,c,j,d)
                                
        return Result
        
        
    def HF_matrix(self,C):
        '''HF-matrix'''
        
        HF_mat = np.empty((self.N,self.N))
        
        for i in range(self.N):
            for j in range(self.N):
                HF_mat[i,j] = HF.HF_elements(self,i,j,C)
        
        return HF_mat
       
       
    def HF_iter(self,tol=1e-8,max_iter=50):
        '''Solving HF with an iterative scheme'''
        
        C = np.eye(self.N)
        ε = np.ones(self.N)
        
        for i in range(max_iter):
            HF_mat = HF.HF_matrix(self,C)
            ε_new, C = np.linalg.eigh(HF_mat)
            C = C.T
            
            if abs(ε[0]-ε_new[0]) < tol:
                break
            ε = ε_new
        
        return HF.calc_E(self,C), i
        
        
    def calc_E(self,C):
        '''Calculate energy'''
        
        E = 0
        for p in range(self.n):
          for a in range(self.N):
            for b in range(self.N):
              E += C[p,a]*C[p,b]*self.elem.OBME(a,b)
              for q in range(self.n):
                for c in range(self.N):
                  for d in range(self.N):
                    E += 0.5*C[p,a]*C[q,b]*C[p,c]*C[q,d]*self.elem.AS(a,b,c,d)
                            
        return E
    

if __name__ == '__main__':

    basis = ((0,0), (0,1), (1,0), (1,1), (2,0), (2,1))
    Helium = HF(2,basis)
    
    Helium.HF_iter()
