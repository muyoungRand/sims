#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from qutip import *

# ------ SIMULATION & PLOT SETTINGS ------ #
# Highest Fock state
nMax = 100 

# Size of Wigner function to plot
sizeWigner = 5  #Higher numbers require higher nMax for accurate simulations
xvec = np.linspace(-sizeWigner, sizeWigner, 200)

# QuTip integration settings
options = Options()
options.num_cpus = 5
options.nsteps = 100000

a = tensor(qeye(2), destroy(nMax))
x = tensor(qeye(2), position(nMax))

sX = tensor(sigmax(), qeye(nMax))
sY = tensor(sigmay(), qeye(nMax))
sZ = tensor(sigmaz(), qeye(nMax))
sM = tensor(destroy(2), qeye(nMax))

pulse_seq = [-0.353282, 0.902028, -0.518804, 
            -0.447977, 0.447404, 0.231963,
            -0.307409, -0.191352, 0.354135,
            0.141061, -0.544457, -0.310069,
            0.685625, 0.623302, -0.144266,
            -0.808851, -0.102101, -0.682909,
            1.021610, 0.534188, -0.54781]
#%%
def plotter(rho, title):
    motion = ptrace(rho, 1)
    qubit = ptrace(rho, 0)
    eigenstate1 = [np.round(qubit.eigenstates()[1][0][0][0][0],3), np.round(qubit.eigenstates()[1][0][1][0][0],3)]
    eigenstate2 = [np.round(qubit.eigenstates()[1][1][0][0][0],3), np.round(qubit.eigenstates()[1][1][1][0][0],3)]

    W = wigner(motion, xvec, xvec)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.vlines(0, ymin = -3, ymax = 3, colors = 'r', linestyles = 'dashed')
    ax.hlines(0, xmin = -3, xmax = 3, colors = 'r', linestyles = 'dashed')

    ax.set_ylabel("Position, x")
    ax.set_xlabel("Momentum, p")

    im = ax.contourf(xvec, xvec, W, 100, norm=mpl.colors.Normalize(-.25,.25), cmap=plt.get_cmap('RdBu'))
    ax.text(-4.0, 4.0, eigenstate1)
    ax.text(-4.0, 3.5, eigenstate2)
    ax.set_title(title)
    
    return motion, qubit

def hamiltonian(rabi_freq, phase_RSB, phase_BSB):
    p = phase_RSB + phase_BSB
    m = phase_RSB - phase_RSB
    
    part1 = sM.dag() * np.exp(1j * p) + sM * np.exp(-1j * p)
    part2 = a * np.exp(1j * m) + a.dag() * np.exp(-1j * m)
    
    return rabi_freq * part1 * part2

rabi_freq = 0.5

phase_BSB = np.pi/2
phase_RSB = np.pi/2

r0 = tensor(basis(2, 0) + basis(2, 1), basis(nMax, 0))

H1 = pulse_seq[0] * sY
H2 = hamiltonian(rabi_freq, phase_RSB, phase_BSB)

r1 = mesolve(H = H1, rho0 = r0, tlist = [0, 1], options = options).states[-1]
m1, q1 = plotter(r1, title = 'After C1')

r2 = mesolve(H = H2, rho0 = r1, tlist = [0, 1], options = options).states[-1]
m2, q2 = plotter(r2, title = 'After C1 + D1')

# %%
def func(x):
    rabi_freq = x[0]
    phase_RSB = x[1]
    phase_BSB = x[2]
    
    r0 = tensor(basis(2, 0) + basis(2, 1), basis(nMax, 0))
    H = hamiltonian(rabi_freq, phase_RSB, phase_BSB)
    
    r = mesolve(H = H, rho0 = r0, tlist = [0, 1], options = options).states[-1]
    
    m = ptrace(r, 1)
    nBar = np.sum([[i * np.diag(np.real(m))[i] for i in range(nMax)]])
    
    q = ptrace(r, 0)
    eig1 = np.real(q.eigenstates()[1][0][0][0][0])
    eig2 = np.real(q.eigenstates()[1][0][1][0][0])
    
    return nBar - 0.25

from scipy.optimize import minimize

bnds = ( (0.0, 2.0), (0, 2 * np.pi), (0, 2* np.pi) )

res = minimize(func, (0.5, 0, np.pi), bounds=bnds, tol = 1E-6)
print(res)
# %%
