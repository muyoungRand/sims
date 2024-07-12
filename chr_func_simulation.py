#%%
import numpy as np
import matplotlib.pyplot as plt

from qutip import *

options = Options()
options.num_cpus = 5
options.nsteps = 1000

# Max Fock state for calculations
N = 20

# Operators
def rotation(angle):
    part1 = np.cos(angle/2) * tensor(qeye(2), qeye(N))
    part2 = -1j * np.sin(angle/2) * tensor(sigmay(), qeye(N))
    return part1 + part2

a = tensor(qeye(2), destroy(N))
sm = tensor(destroy(2), qeye(N))

def displacement(rsb_phase = 0, bsb_phase = 0):
    RSB = a * sm.dag() * np.exp(1j * rsb_phase)
    RSBp = a.dag() * sm * np.exp(-1j * rsb_phase)

    BSB = a.dag() * sm.dag() * np.exp(1j * bsb_phase)
    BSBp = a * sm * np.exp(-1j * bsb_phase)

    return [RSB, RSBp, BSB, BSBp]

#%%
# Test States
#psi0 = tensor(ket2dm(basis(2,1)), ket2dm(basis(N, 3))) 
#psi0 = tensor(ket2dm(basis(2,1)), coherent_dm(N, 5))

# Cubic Phase State
ideal_x = position(N)
ideal_H = ideal_x**3
ideal_psi0 = basis(N)
ideal_output = mesolve(H = ideal_H, rho0 = ideal_psi0, tlist = [0.0, 1.0], options = options)
ideal_rho = ideal_output.states[-1] # Cubic Phase State with Sqz Param = 1

# Perform Chr Function Measurement
psi0 = tensor(basis(2,0), ideal_rho) # Add the spin part (assumed to be in ground state)
psi1 = rotation(np.pi/2) * psi0

nGrid = 100
amp = np.linspace(0, 3, nGrid) # Amplitude of displacement operation

#%%
# Upper quadrant
x = np.linspace(0, 3, nGrid)
y = np.linspace(0, 3, nGrid)
Z = np.zeros((nGrid, nGrid))

bsb_phase = np.linspace(0, np.pi, nGrid)

for i in range(len(bsb_phase)):    
    for j in range(len(amp)):
        output = mesolve(H = displacement(0, bsb_phase[i]), rho0 = psi1, tlist = [0, amp[j]])
        state = output.states[-1]

        ion_state = ptrace(state, 0)
        excited_state = 1 - 2*np.absolute(expect(sigmaz(), ion_state))

        real = np.cos(bsb_phase[i]/2)
        img = np.sin(bsb_phase[i]/2)

        check_real = np.absolute(x - amp[j]*real)
        check_img = np.absolute(x - amp[j]*img)

        loc_x = np.where(check_real == min(check_real))[0]
        loc_y = np.where(check_img == min(check_img))[0]

        Z[loc_x, loc_y] = excited_state

#%%
# Lower quadrant
x = np.linspace(0, 3, nGrid)
yneg = np.linspace(0, -3, nGrid)
Zneg = np.zeros((nGrid, nGrid))

bsb_phase = np.linspace(0, -np.pi, nGrid)

for i in range(len(bsb_phase)):    
    for j in range(len(amp)):
        output = mesolve(H = displacement(0, bsb_phase[i]), rho0 = psi1, tlist = [0, amp[j]])
        state = output.states[-1]

        ion_state = ptrace(state, 0)
        excited_state = 1 - 2*np.absolute(expect(sigmaz(), ion_state))

        real = np.cos(bsb_phase[i]/2)
        img = np.sin(bsb_phase[i]/2)

        check_real = np.absolute(x - amp[j]*real)
        check_img = np.absolute(amp[j]*img - yneg)

        loc_x = np.where(check_real == min(check_real))[0]
        loc_y = np.where(check_img == min(check_img))[0]

        Zneg[loc_x, loc_y] = excited_state
#%%       
# Plot everything together 
levels = np.linspace(-1, 1, 101)

h = plt.contourf(x, np.linspace(3, -3, 2*nGrid), np.vstack((np.flip(Z,0), Zneg)), 50, levels = levels)
plt.axis('scaled')
plt.colorbar()
plt.show()

# %%
