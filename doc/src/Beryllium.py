import numpy as np
from CIS import *
from Hartree_Fock import *

basis = ((0,0), (0,1), (1,0), (1,1), (2,0), (2,1))
Beryllium = CIS(4,basis)
Beryllium_HF = HF(4,basis)

# === Configuration Interaction Singles ===
A = np.zeros((5,5))

# --- <c|H|c> ---
A[0,0] = Beryllium.c_H_c()

# --- <c|H|p_i^a> ---
A[0,1] = Beryllium.c_H_ia(0,4)
A[0,2] = Beryllium.c_H_ia(1,5)
A[0,3] = Beryllium.c_H_ia(2,4)
A[0,4] = Beryllium.c_H_ia(3,5)

# --- <p_i^a|H|c> ---
A[1,0] = Beryllium.c_H_ia(0,4)
A[2,0] = Beryllium.c_H_ia(1,5)
A[3,0] = Beryllium.c_H_ia(2,4)
A[4,0] = Beryllium.c_H_ia(3,5)

# --- <p_i^a|H|p_j^b> ---
# <12|H|21>
A[1,1] = Beryllium.ia_H_jb(0,4,0,4)
A[1,2] = Beryllium.ia_H_jb(0,4,1,5)
A[2,1] = Beryllium.ia_H_jb(1,5,0,4)
A[2,2] = Beryllium.ia_H_jb(1,5,1,5)

# <12|H|31>
A[1,3] = Beryllium.ia_H_jb(0,4,2,4)
A[1,4] = Beryllium.ia_H_jb(0,4,3,5)
A[2,3] = Beryllium.ia_H_jb(1,5,2,4)
A[2,4] = Beryllium.ia_H_jb(1,5,3,5)

# <13|H|21>
A[3,1] = Beryllium.ia_H_jb(2,4,0,4)
A[3,2] = Beryllium.ia_H_jb(3,5,0,4)
A[4,1] = Beryllium.ia_H_jb(1,5,2,4)
A[4,2] = Beryllium.ia_H_jb(1,5,3,5)

# <13|H|31>
A[3,3] = Beryllium.ia_H_jb(2,4,2,4)
A[3,4] = Beryllium.ia_H_jb(2,4,3,5)
A[4,3] = Beryllium.ia_H_jb(3,5,2,4)
A[4,4] = Beryllium.ia_H_jb(3,5,3,5)

eigvals, eigvecs = np.linalg.eigh(A)

print(A)

print('Ref. Energy: ', A[0,0])
print('Energy CIS:  ', eigvals[0])

# === Hartree Fock ===
#print('Energy HF:   ', Beryllium_HF.HF_iter()[0],' in ', Beryllium_HF.HF_iter()[1], ' iterations')
