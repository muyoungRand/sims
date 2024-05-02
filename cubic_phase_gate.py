#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from qutip import *

from gen_cubic_phase_sequence import generate_cubic_sequence as generator

"""
Part 1 - Generate pulse sequence

    Free Parameters:
        1) Pulse Time (t) = How much to displace during Rabi interaction (sigma * X interaction)
        2) Pulse Number (N) = How many pulses/Fourier coefficients
        3) Displacement = Amplitute of final displacement pulse

    Supposed to be free but NOT free parameters:
        1) Squeezing parameter - Assumed to be 1 throughout.
        (This is due to the Airy function approximation only working for Chi = 1)
"""

# Experimentally, corresponds to time for displacement of alpha = 0.5
pulse_time = 0.5

pulse_number = 9
displacement = 2

pulse_seq = generator(pulse_time, pulse_number, displacement)
""" pulse_seq = [-0.353282, 0.902028, -0.518804, 
            -0.447977, 0.447404, 0.231963,
            -0.307409, -0.191352, 0.354135,
            0.141061, -0.544457, -0.310069,
            0.685625, 0.623302, -0.144266,
            -0.808851, -0.102101, -0.682909,
            1.021610, 0.534188, -0.54781] """

"""
Part 2 - Simulation of Pulse Sequence Results

    Compare output Wigner functions of Ideal vs. Pulse Sequence vs. Trapped Ion implementation.

    Additionally, there is one free parameter, the final displacement pulse, which is optimised.
"""
# Highest Fock state
nMax = 200 

# Size of Wigner function to plot
sizeWigner = 5  #Higher numbers require higher nMax for accurate simulations
xvec = np.linspace(-sizeWigner, sizeWigner, 200)

# QuTip integration settings
options = Options()
options.num_cpus = 5
options.nsteps = 100000


# ---------- Ideal Cubic Phase Gate ---------- #
ideal_x = position(nMax)
ideal_H = ideal_x**3

ideal_psi0 = basis(nMax) # Initialise in |n=0> state
t = [0.0, 1.0]     # We don't care about the internal dynamics, just the start and end results

ideal_output = mesolve(H = ideal_H, rho0 = ideal_psi0, tlist = t, options = options)
ideal_rho = ideal_output.states[-1]
ideal_W = wigner(ideal_rho, xvec, xvec)


# ---------- Rabi Gate-based Phase Gate ---------- #
a = tensor(qeye(2), destroy(nMax))
x = tensor(qeye(2), position(nMax))
sY = tensor(sigmay(), qeye(nMax))
sZ = tensor(sigmaz(), qeye(nMax))
sM = tensor(destroy(2), qeye(nMax))

def pulse(initial_state, angle, pulse_time):
    """
    Actual of a single pulse 
        First, perform spin-only rotation based on optimal pulse sequence
        Second, perform Rabi interaction for pulse_time
        Repeat for (pulse_number + 1) times
    """
    t = [0.0, 1.0]

    H1 = - angle * sY
    H2 = - pulse_time * sZ * x

    output = mesolve(H = H1, rho0 = initial_state, tlist = t, options = options)
    rho1 = output.states[-1]

    output2 = mesolve(H = H2, rho0 = rho1, tlist = t, options = options)
    rho2 = output2.states[-1]

    return rho2

pulse_rho = tensor(basis(2,0), basis(nMax, 0))

for i in pulse_seq:
    pulse_rho = pulse(pulse_rho, i, pulse_time) # Iterative evolution


def find_best_displacement(state, optimise_displacement, ideal_rho):
    """
    Try differnt final displacement amplitudes, and return the one that has highest fidelity
    """
    ls = np.arange(0, 2 * optimise_displacement, 0.1)
    fidel = []

    for i in ls:
        displace_H = i * x

        output = mesolve(H = displace_H, rho0 = state, tlist = [0.0, 1.0], options = options)
        output_motion = ptrace(output.states[-1], 1)

        fidel.append(fidelity(ideal_rho, output_motion))

    best = ls[np.where(fidel == np.max(fidel))[0][0]]

    return best

pulse_best_displacement = find_best_displacement(pulse_rho, displacement, ideal_rho)

displace_H = pulse_best_displacement * x

pulse_output = mesolve(H = displace_H, rho0 = pulse_rho, tlist = [0.0, 1.0], options = options)
pulse_rho_motion = ptrace(pulse_output.states[-1], 1)

pulse_W = wigner(pulse_rho_motion, xvec, xvec)


# ---------- Trapped Ion Implementation ---------- #
# NOTE: Use beyond-Lamb Dicke Regime Hamiltonian since resulting Wigner function is big

ld_param = 0.2
rabi_freq = (np.pi / (2 * pulse_time)) * ld_param * np.exp(-0.5 * (ld_param**2)) # Rquired Experimental Value

def hamiltonian(order, ld_param, sideband):
    factor = (1j * ld_param) ** (2 * order)
    denom = math.factorial(order) * math.factorial(order + 1)

    if sideband == 'r':
        operator = sM.dag() * (a.dag()**order) * (a**(order + 1))
    elif sideband == 'b':
        operator = sM.dag() * (a.dag()**(order + 1)) * (a**order)
    else:
        print("Use either 'r' for RSB or 'b' for BSB")

    return (factor / denom) * operator + (np.conjugate(factor / denom)) * operator.dag()

RSB_H = rabi_freq * (hamiltonian(0, ld_param, 'r') + hamiltonian(1, ld_param, 'r') + hamiltonian(2, ld_param, 'r'))
BSB_H = rabi_freq * (hamiltonian(0, ld_param, 'b') + hamiltonian(1, ld_param, 'b') + hamiltonian(2, ld_param, 'b'))

def pulse_ion(initial_state, angle, pulse_time):
    t = [0.0, 1.0]

    H1 = - angle * sY
    H2 = - pulse_time * (RSB_H + BSB_H)

    output = mesolve(H = H1, rho0 = initial_state, tlist = t, options = options)
    rho1 = output.states[-1]

    output2 = mesolve(H = H2, rho0 = rho1, tlist = t, options = options)
    rho2 = output2.states[-1]

    return rho2

ion_rho = tensor( (1 / np.sqrt(2)) * (basis(2, 0) + basis(2, 1)), basis(nMax, 0)) # Need to start in SigmaX basis

for i in pulse_seq:
    ion_rho = pulse_ion(ion_rho, i, pulse_time) # Iterative evolution

ion_best_displacement = find_best_displacement(ion_rho, displacement, ideal_rho)


displace_H = ion_best_displacement * x

ion_output = mesolve(H = displace_H, rho0 = ion_rho, tlist = [0.0, 1.0], options = options)
ion_rho_motion = ptrace(ion_output.states[-1], 1)

ion_W = wigner(ion_rho_motion, xvec, xvec)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for i in range(len(ax)):
    ax[i].vlines(0, ymin = -3, ymax = 3, colors = 'r', linestyles = 'dashed')
    ax[i].hlines(0, xmin = -3, xmax = 3, colors = 'r', linestyles = 'dashed')

im = ax[0].contourf(xvec, xvec, ideal_W, 100, norm=mpl.colors.Normalize(-.25,.25), cmap=plt.get_cmap('RdBu'))
ax[0].set_title("Ideal Cubic Phase")

im = ax[1].contourf(xvec, xvec, pulse_W, 100, norm=mpl.colors.Normalize(-.25,.25), cmap=plt.get_cmap('RdBu'))
ax[1].text(-4.0, 4.5, "Pulse ideal displacment = " + str(np.round(pulse_best_displacement, 3)))
ax[1].text(-4.0, 4.0, "Fidelity (Ideal vs Pulse) = " + str(np.round(fidelity(ideal_rho, pulse_rho_motion), 3)), verticalalignment = 'center')
ax[1].set_title("Rabi Gate Sequence")

im = ax[2].contourf(xvec, xvec, ion_W, 100, norm=mpl.colors.Normalize(-.25,.25), cmap=plt.get_cmap('RdBu'))
ax[2].text(-4.0, 4.5, "Ion ideal displacment = " + str(np.round(ion_best_displacement, 3)))
ax[2].text(-4.0, 4.0, "Fidelity (Ideal vs Ion) = " + str(np.round(fidelity(ideal_rho, ion_rho_motion), 3)), verticalalignment = 'center')
ax[2].text(-4.0, 3.5, "Fidelity (Pulse vs Ion) = " + str(np.round(fidelity(pulse_rho_motion, ion_rho_motion), 3)), verticalalignment = 'center')
ax[2].set_title("Trapped Ion Implementation")

fig.colorbar(im, ax = ax.ravel().tolist())

# %%
