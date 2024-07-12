#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from qutip import *

options = Options()
options.num_cpus = 5
options.nsteps = 100000

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
# Random Testing States
#psi0 = tensor(ket2dm(basis(2,1)), ket2dm(basis(N, 3))) 
#psi0 = tensor(ket2dm(basis(2,1)), coherent_dm(N, 5))

# Prepare Cubic Phase State
ideal_x = position(N)
ideal_H = ideal_x**3
ideal_psi0 = basis(N) # Initialise in |n=0> state
t = [0.0, 1.0]     # We don't care about the internal dynamics, just the start and end results
ideal_output = mesolve(H = ideal_H, rho0 = ideal_psi0, tlist = t, options = options)
ideal_rho = ideal_output.states[-1]
ideal_rho = tensor(basis(2,0), ideal_rho)

# Perform Chr Function Measurement
psi1 = rotation(np.pi/2) * ideal_rho

nGrid = 100
amp = np.linspace(0, 3, nGrid) # Amplitude of displacement operation

x = np.linspace(0, 3, nGrid)
y = np.linspace(0, 3, nGrid)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

bsb_phase = np.linspace(0, np.pi, nGrid)

for i in range(len(bsb_phase)):    
    for j in range(len(amp)):
        output = mesolve(H = displacement(0, bsb_phase[i]), rho0 = psi1, tlist = [0, amp[j]])
        state = output.states[-1]

        ion_state = ptrace(state, 0)
        excited_state = np.absolute(expect(sigmaz(), ion_state))

        real = np.cos(bsb_phase[i]/2)
        img = np.sin(bsb_phase[i]/2)

        check_real = np.absolute(x - amp[j]*real)
        check_img = np.absolute(x - amp[j]*img)

        loc_x = np.where(check_real == min(check_real))[0]
        loc_y = np.where(check_img == min(check_img))[0]

        Z[loc_x, loc_y] = excited_state#
#%%        
levels = np.linspace(-0.01, 1.01, 101)

h = plt.contourf(x, y, Z, 50, levels = levels)
plt.axis('scaled')
plt.colorbar()
plt.show()

# %%
# ---------- Calibrate displacement amplitude vs. displacement duration ---------- #
""" 
psi0 = tensor(ket2dm(basis(2,0)), ket2dm(basis(N, 0))) 

t = np.linspace(0, 1, 101)
output = mesolve(H = displacement(), rho0 = psi0, tlist = t, e_ops = [a.dag() * a, sm.dag() * sm])
n_c = output.expect[0]
n_a = output.expect[1]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(t, n_a, 'r-', label="Excited Population, P(e)")
axes[0].plot(t, np.sqrt(n_c), 'b-', label="Displacement Amplitude, alpha")
axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
axes[0].set_ylim([0, 1])
axes[0].legend()
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Excitation probability")

output = mesolve(H = H, rho0 = psi0, tlist = t)
rho_check = output.states[-1]

xvec = np.linspace(-3, 3, 200)
rho_phonon_check = ptrace(rho_check, 1)
W = wigner(rho_phonon_check, xvec, xvec)

im = axes[1].contourf(xvec, xvec, W, 100, norm=mpl.colors.Normalize(-.25,.25), cmap=plt.get_cmap('RdBu'))
axes[1].vlines(0, ymin = -3, ymax = 3, colors = 'r', linestyles = 'dashed')
axes[1].hlines(0, xmin = -3, xmax = 3, colors = 'r', linestyles = 'dashed')
axes[1].set_title("Wigner Function of final phonon state")
plt.colorbar(im)
 """
# %%
