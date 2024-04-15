#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
import scipy.special as special
# %%
mode_frequency = 2 * np.pi * 500 * 10**(3)
wavelength = 355 * 10**(-9)
raman_angle = 90 #Degrees
mass = 171 * constants.atomic_mass
eta = 2 * np.sin((raman_angle * np.pi/180)/2) * (2 * np.pi / wavelength) * np.sqrt(constants.hbar / (2 * mass * mode_frequency))

print("Lamb-Dicke Parameter:", np.round(eta, 3))

nBar = 3
nMax = 150
rabi_freq = 1 #100us RSB pi-time

#%%
def eval_rabi_freq(nStart, nDelta, eta = eta):
    """
    Calculates Rabi Frequency for nStart -> nStart + nDelta

    Args:
        nStart (int): Initial phonon state
        nDelta (int): Change in phonon number
        eta (float): Lamb-Dicke Parameter

    Returns:
        float: Rabi frequency
    """
    nEnd = nStart + nDelta

    if nEnd < 0:
        return 0
    else:
        nSmall = min([nStart, nEnd])
        nBig = max([nStart, nEnd])
        factor2 = np.exp(-0.5 * eta**2) * eta**(np.absolute(nDelta))
        factor3 = np.sqrt(np.math.factorial(nSmall)/np.math.factorial(nBig))
        factor4 = special.assoc_laguerre(eta**2, nSmall, np.absolute(nDelta))
    return factor2 * factor3 * factor4

def thermal_distribution(n, nBar = nBar):
    return (nBar**n) / ((nBar + 1)**(n+1))

def a_n(t, n):
    return np.cos(rabi_freq * eval_rabi_freq(n, -1) * t / 2)**2

def b_n(t, n):
    return np.sin(rabi_freq * eval_rabi_freq(n, -1) * t / 2)**2

def W(t, dims = nMax):
    matrix = np.zeros([dims, dims])

    matrix[0, 0] = 1
    matrix[0, 1] = b_n(t, 1)
    for i in range(1, len(matrix)):
        matrix[i, i] = a_n(t, i)
        if i < nMax-1:
            matrix[i, i+1] = b_n(t, i)

    return matrix
#%%
'''
Find phonon transition matrix elements for the Raman beams.
Use this to find the phonon-trapping states: the phonon numbers at which a specific n-th order phonon transition goes to 0.

Source: 10.1103/PhysRevA.104.043108
'''
nDelta = [-1, -2, -3]
nStart = [1, 2, 3]
n_scan = [np.arange(i, 150, 1) for i in nStart]
rabi_list = [[] for i in range(len(nDelta))]

for i in range(len(nDelta)):
    n = n_scan[i]
    for k in n:
        res = np.absolute(eval_rabi_freq(k, nDelta[i]))
        rabi_list[i].append(res)

first_order_phonon_trap = n_scan[0][np.argmin(rabi_list[0])]

#%%
"""
Evaluate classic pulsed SBC 

    Starting from some n (determined by classic_n_start)
    Run pi-pulses for n-th harmonic level down to ground state
"""
doppler_population = [thermal_distribution(i) for i in range(nMax)]
classic_method = [(eval_rabi_freq(i, -1)) for i in range(nMax)]

nPulses = np.arange(2, 50, 1)

res = []
res2 = []
for i in nPulses:
    classic_W_list = []
    classic_time_list = []
    for j in range(i):
        time = np.pi/classic_method[i - j]
        classic_W_list.append(W(time))
        classic_time_list.append(time)

    classic_W_eff = np.linalg.multi_dot(classic_W_list)

    classic_total_time = np.sum(classic_time_list)
    classic_final_population = np.dot(classic_W_eff, doppler_population)
    classic_final_population = classic_final_population / np.sum(classic_final_population)
    classic_final_nBar = np.sum(i * classic_final_population[i] for i in range(nMax))
    res.append(classic_final_nBar)
    res2.append(classic_final_population[0])

plt.plot(res, label = 'nBar')
plt.plot(res2, label = 'P(n=0)')
plt.title("Classic SBC")
plt.xlabel("Number of Pulses")
plt.legend()

#%%
optimal_nPulses = 30

classic_W_list = []
classic_time_list = []
for i in range(optimal_nPulses):
    time = np.pi/classic_method[optimal_nPulses - i]
    classic_W_list.append(W(time))
    classic_time_list.append(time)

classic_W_eff = np.linalg.multi_dot(classic_W_list)

classic_total_time = np.sum(classic_time_list)
classic_final_population = np.dot(classic_W_eff, doppler_population)
classic_final_population = classic_final_population / np.sum(classic_final_population)

#%%
"""
Fixed, single-order protocol

    Here, pulse duration of each cooling pulse is same.
    Find optimal pulse duration for maximum cooling.
"""
N = optimal_nPulses # Number of pulses

def fixed_single_func(t0):
    W_eff = np.linalg.multi_dot([W(t0) for i in range(N)])
    pop = np.dot(W_eff, doppler_population)

    return np.sum([i * pop[i] for i in range(len(pop))])

test = np.arange(0.0, 10, 0.1)
test_res = []
for i in test:
    test_res.append(fixed_single_func(i))

best = test[np.argmin(test_res)]
fixed_single_W_eff = np.linalg.multi_dot([W(np.absolute(best)) for i in range(N)])
fixed_single_final_population = np.dot(fixed_single_W_eff, doppler_population)
fixed_single_final_population = fixed_single_final_population / np.sum(fixed_single_final_population)

# %%
fig, ax = plt.subplots(1, 2, figsize = (10, 5))

for i in range(len(nDelta)):
    ax[0].plot(n_scan[i], rabi_list[i], label = 'RSB Order ' + str(nDelta[i]))
ax[0].annotate(first_order_phonon_trap, (first_order_phonon_trap , 0.01), textcoords = "offset points", xytext = (0, 30), ha = 'center', arrowprops = dict(facecolor='black', shrink=0.01))
ax[0].set_xlabel('Phonon State n')
ax[0].set_ylabel('Normalised Rabio Frequency')
ax[0].legend()

ax[1].set_yscale('log')
ax[1].set_ylim([min(fixed_single_final_population), 1])
ax[1].set_xlabel('Phonon state n')
ax[1].set_ylabel('Occupation Probability')
ax[1].plot(doppler_population, label = "Doppler Population")
ax[1].plot(classic_final_population, label = "Classic SBC Population")
ax[1].plot(fixed_single_final_population, label = "Fixed Single Order SBC Population")
ax[1].legend()

plt.tight_layout()

print("1st Order RSB rabi frequency 0-crossing at n =", first_order_phonon_trap)
print("For nBar = ", nBar, "and number of pulses = ", optimal_nPulses)
print("P(n>2) after Classic SBC = ", np.round(np.sum(classic_final_population[2:]), 3))

print("Doppler Population: ", np.round(doppler_population[0:5], 3))
print("Classic SBC Population: ", np.round(classic_final_population[0:5], 3))
print("Fixed First Order SBC Population: ", np.round(fixed_single_final_population[0:5], 3))
# %%
