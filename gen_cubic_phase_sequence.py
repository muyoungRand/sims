#%%
import numpy as np
from itertools import product
from scipy import special
from scipy.optimize import minimize

def airy_approx(t, n):
    """
    Approximation solution to space intgeral

    NOTE: Error in paper - Missing factor of 1/sqrt(pi) on RHS
    NOTE: This ONLY works for squeezing parameter of 1; not sure to generalise for other squeeze values. Might need to just perform the integration.
    """
    beta = t * n
    return 4.691 * np.exp(-beta/3) * special.airy(0.231 - 0.693 * beta)[0] / np.sqrt(np.pi)

def _term_(j, sj_list, phi_list):
    """
    Generate each term within the product
    """
    power = (sj_list[j+1] - sj_list[j]) / 2

    numerator = np.exp(1j * phi_list[j]) + (-1)**(power**2) * np.exp(-1j * phi_list[j])
    denominator = 2 * 1j**power

    return numerator/denominator

def func(phi_list, t, full_list, displacement):
    """
    Calculates infidelity given a list of angles (phi_list)

    Is also the function to minimise to obtain the optimal list of angles
    """
    components = []
    exponents = []

    for ls in full_list:
        inter = []

        for i in range(len(ls) - 1):
            inter.append(_term_(i, ls, phi_list))

        components.append(np.prod(inter))
        exponents.append(np.sum(ls[1:-1]) + displacement)
        
    approx_integral = [airy_approx(t, i) for i in exponents]

    results = []
    for i in range(len(components)):
        results.append(components[i] * approx_integral[i])

    return 1.0 - np.absolute(np.sum(results))**2

def generate_cubic_sequence(pulse_time, pulse_number, displacement):

    t = pulse_time
    N = pulse_number

    possible_sj = [-1, +1]

    full_list = [list(p) for p in product(possible_sj, repeat = N)]
    for i in full_list:
        i.insert(0, -1) #s0 = -1 
        i.append(-1) #sN+1 = -1

    root = minimize(func, [np.random.uniform(-np.pi/2, np.pi/2) for i in range(N+1)], args = (t, full_list, displacement))
    print(root.message)
    print("For Displacement = ", np.round(displacement, 3))
    print("Solution for Optimal Angles: ", np.array(root.x))
    print("Calculated infidelity: ", np.round(func(root.x, t, full_list, displacement), 3), '\n')

    return np.array(root.x)

# %%
# Backup scripts for testing.
# To Show Error 1 
""" 
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def exact(x, beta):
    return (1/np.sqrt(np.pi)) * np.exp(-x**2) * np.exp(-1j * 1 * x**3) * np.exp(1j * beta * x)

def approx(beta):
    return 4.691 * np.exp(-beta/3) * special.airy(0.231 - 0.693 * beta)[0] / np.sqrt(np.pi)

trial = [0.1*i for i in range(1,11)]

res_exact = []
res_approx = []
res_approx2 = []
for k in trial:
    res_exact.append(integrate.quad(exact, -np.inf, np.inf, args = k)[0])
    res_approx.append(approx(k))

plt.plot(trial, res_exact, label = 'Numerical Integration')
plt.plot(trial, res_approx, label = 'Approximate Airy Function with 1/np.sqrt(pi)')
plt.legend()  """

# Generate Fourier Coefficients
"""
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def integrand(x, index, time):
    return np.exp(1j * 1 * x**3) * np.exp(-1j * index * time * x)

def coeff(time, index):
    norm = time / (2 * np.pi)
    res = np.real(integrate.quad(integrand, -np.pi/time, np.pi/time, args = (index, time))[0])
    return norm * res

check_N = 20
check_time = 0.5

val_coeff = []
index_coeff = np.arange(-check_N/2, (check_N/2) + 1, 1)
for i in index_coeff:
    val_coeff.append(coeff(check_time, i))

plt.plot(index_coeff, np.absolute(val_coeff), 'x-', label = "N = " + str(check_N) + ", Time = " + str(np.round(check_time, 3)))
plt.ylabel("Fourier Amplitude (Absolute Value)")
plt.xlabel("Fourier Index")
plt.legend()
plt.show() 
"""