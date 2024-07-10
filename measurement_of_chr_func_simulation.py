#%%
import numpy as np
import matplotlib.pyplot as plt

from qutip import *

# Max Fock state for calculations
N = 10

# Operators
def rotation(angle):
    part1 = np.cos(angle/2) * tensor(qeye(2), qeye(N))
    part2 = -1j * np.sin(angle/2) * tensor(sigmay(), qeye(N))
    return part1 + part2

#   Displacement
def displacement(alpha):
    return tensor(sigmax(), displace(N, alpha))

psi0 = tensor(ket2dm(basis(2, 1)), coherent_dm(N, 0))
# %%
# Real Part of Characteristic Function
psi1 = rotation(0) * psi0
psi2 = displacement(1.0) * psi1

# %%
