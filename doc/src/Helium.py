import numpy as np
from CIS import *
from Hartree_Fock import *

basis = ((0,0), (0,1), (1,0), (1,1), (2,0), (2,1))

Helium_HF = HF(2,basis)
Helium = CIS(2,basis)

# === Configuration Interaction Singles ===
A = np.zeros((5,5))

# --- <c|H|c> ---
A[0,0] = Helium.c_H_c()

# --- <c|H|p_i^a> ---
A[0,1] = Helium.c_H_ia(0,2)
A[0,2] = Helium.c_H_ia(1,3)
A[0,3] = Helium.c_H_ia(0,4)
A[0,4] = Helium.c_H_ia(1,5)

# --- <p_i^a|H|c> ---
A[1,0] = Helium.c_H_ia(0,2)
A[2,0] = Helium.c_H_ia(1,3)
A[3,0] = Helium.c_H_ia(0,4)
A[4,0] = Helium.c_H_ia(1,5)

# --- <p_i^a|H|p_j^b> ---
# <12|H|21>
A[1,1] = Helium.ia_H_jb(0,2,0,2)
A[1,2] = Helium.ia_H_jb(0,2,1,3)
A[2,1] = Helium.ia_H_jb(1,3,0,2)
A[2,2] = Helium.ia_H_jb(1,3,1,3)

# <12|H|31>
A[1,3] = Helium.ia_H_jb(0,2,0,4)
A[1,4] = Helium.ia_H_jb(0,2,1,5)
A[2,3] = Helium.ia_H_jb(1,3,0,4)
A[2,4] = Helium.ia_H_jb(1,3,1,5)

# <13|H|21>
A[3,1] = Helium.ia_H_jb(0,4,0,2)
A[3,2] = Helium.ia_H_jb(1,5,0,2)
A[4,1] = Helium.ia_H_jb(0,4,1,3)
A[4,2] = Helium.ia_H_jb(1,5,1,3)

# <13|H|31>
A[3,3] = Helium.ia_H_jb(0,4,0,4)
A[3,4] = Helium.ia_H_jb(0,4,1,5)
A[4,3] = Helium.ia_H_jb(1,5,0,4)
A[4,4] = Helium.ia_H_jb(1,5,1,5)

print(A)

eigvals, eigvecs = np.linalg.eigh(A)

print('Ref. Energy: ', A[0,0])
print('Energy CIS:  ', eigvals[0])

# === Hartree Fock ===
print('Energy HF:   ', Helium_HF.HF_iter()[0],' in ', Helium_HF.HF_iter()[1], ' iterations')
