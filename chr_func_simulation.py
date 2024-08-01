#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import cm

from qutip import *

options = Options()
options.num_cpus = 5
options.nsteps = 1000

# Max Fock state for calculations
N = 20

# Operators
def rotation(angle):
    part1 = np.cos(angle/2) * tensor(qeye(2), qeye(N))
    part2 = -1j * np.sin(angle/2) * tensor(sigmax(), qeye(N))
    return part1 + part2

a = tensor(qeye(2), destroy(N))
sm = tensor(destroy(2), qeye(N))

def displacement(rsb_phase = 0, bsb_phase = 0):
    RSB = a * sm.dag() * np.exp(1j * rsb_phase)
    RSBp = a.dag() * sm * np.exp(-1j * rsb_phase)

    BSB = a.dag() * sm.dag() * np.exp(1j * bsb_phase)
    BSBp = a * sm * np.exp(-1j * bsb_phase)

    return [RSB, RSBp, BSB, BSBp]

# Test States
#psi0 = tensor(ket2dm(basis(2, 0)), ket2dm(basis(N, 1))) 
#psi0 = tensor(ket2dm(basis(2,1)), coherent_dm(N, 5))

# Cubic Phase State
ideal_x = position(N)
ideal_H = ideal_x**3
ideal_psi0 = basis(N, 0)
ideal_output = mesolve(H = ideal_H, rho0 = ideal_psi0, tlist = [0.0, 1.0], options = options)
ideal_rho = ideal_output.states[-1] # Cubic Phase State with Sqz Param = 1 
psi0 = tensor(basis(2, 0), ideal_rho)

psi0 = tensor(basis(2, 0), basis(N, 1))

# Check if generated state is correct -> Plot Wigner function
""" xvec = np.linspace(-5,5,200)
W = wigner(ideal_rho, xvec, xvec)
fig, axes = plt.subplots(1, 1, figsize=(6,6))
cont0 = axes.contourf(xvec, xvec, W, 100, cmap=cm.RdBu)
fig.colorbar(cont0)
plt.show()
 """

# For plotting
x_data = np.linspace(0.1, 400, 101)
radius = [i * 0.006184 for i in x_data]

theta = np.linspace(-180, 180, 101)
theta = [i * np.pi/180 for i in theta]

# For plotting & Storing Data
theta_array = [theta for i in range(len(radius))]
radius_array = [[radius[j] for i in range(len(theta))] for j in range(len(radius))]

nColumns = len(theta)
nRows = len(radius)

plot_data = np.zeros((nRows, nColumns))

# Plot settings
vmin = -1
vmax = 1
levels = 20
level_boundaries = np.linspace(vmin, vmax, levels + 1)

#%%
# ---------------------- Real Part ---------------------- #
psi1 = rotation(0) * psi0

plot_data = np.zeros((nRows, nColumns))

res = []

for phase in theta:
    for amp in radius:
        output = mesolve(H = displacement(0, phase), rho0 = psi1, tlist = [0, amp])
        state = output.states[-1]
        
        ion_state = ptrace(state, 0)
        proj = expect(sigmaz(), ion_state)

        res.append(proj)

counter = 0

for i in range(len(res)):
    if i > 1 and i % nRows == 0:
        counter += 1

    plot_data[i - counter * nRows][counter] = res[i]

fig, ax = plt.subplots(subplot_kw = dict(projection='polar'), figsize = (10, 10))
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)

cax = ax.contourf(theta, radius, plot_data)
cbar = fig.colorbar(cax)
fig.suptitle("Real Part", size = 'xx-large')
#%%
""" # ---------------------- Imaginary Part ---------------------- #
psi1 = rotation(np.pi/2) * psi0

plot_data = np.zeros((nRows, nColumns))

res = []

for phase in theta:
    for amp in radius:
        output = mesolve(H = displacement(0, phase), rho0 = psi1, tlist = [0, amp])
        state = output.states[1]
        
        ion_state = ptrace(state, 0)
        proj = np.absolute(expect(sigmaz(), ion_state))

        res.append(proj)

counter = 0

for i in range(len(res)):
    if i > 1 and i % nRows == 0:
        counter += 1

    plot_data[i - counter * nRows][counter] = res[i]

fig, ax = plt.subplots(subplot_kw = dict(projection='polar'), figsize = (10, 10))
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)

cax = ax.contourf(theta, radius, plot_data)
cbar = fig.colorbar(cax)
fig.suptitle("Imaginary Part", size = 'xx-large') """
# %%
