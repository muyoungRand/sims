#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from qutip import *

# ------ SIMULATION & PLOT SETTINGS ------ #
# Highest Fock state
nMax = 20 

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

def plotter(rho, title):
    motion = ptrace(rho, 1)
    nBar = np.sum([i * np.real(np.diag(motion)[i]) for i in range(nMax)])
    W = wigner(motion, xvec, xvec)

    qubit = ptrace(rho, 0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.vlines(0, ymin = -3, ymax = 3, colors = 'r', linestyles = 'dashed')
    ax.hlines(0, xmin = -3, xmax = 3, colors = 'r', linestyles = 'dashed')

    ax.set_ylabel("Position, x")
    ax.set_xlabel("Momentum, p")

    im = ax.contourf(xvec, xvec, W, 100, norm=mpl.colors.Normalize(-.25,.25), cmap=plt.get_cmap('RdBu'))
    ax.text(-4.0, 4.0, np.round(qubit[:],3))
    ax.text(-4.0, -4.0, 'nBar ='+str(np.round(nBar, 3)))
    ax.set_title(title)
    
    return motion, qubit

def displace(phase):
    spin = np.cos(phase) * sX - np.sin(phase) * sY
    motion = x

    return spin * motion
#%%
r0 = tensor(basis(2, 0) + basis(2, 1), basis(nMax, 0))

phase = 2 * np.pi * np.random.rand()
print(np.round((phase * 180/np.pi)/np.pi, 3))

H = displace(phase * 2)
r = mesolve(H = -H, rho0 = r0, tlist = [0, 1], options = options).states[-1]
m, q = plotter(r, title = 'CarrierLess')

H1 = -sY
H2 = sX * x
H3= sY

r1 = mesolve(H = H1, rho0 = r0, tlist = [0, phase], options = options).states[-1]
r2 = mesolve(H = H2, rho0 = r1, tlist = [0, 1], options = options).states[-1]
r3 = mesolve(H = H3, rho0 = r2, tlist = [0, phase], options = options).states[-1]
m3, q3 = plotter(r3, title = 'Ideal')

# %%
