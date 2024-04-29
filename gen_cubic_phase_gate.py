#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy import special
from scipy.optimize import minimize, shgo

t = 0.5 # Arbitrary time units
N = 14 # Number of Fourier coefficients
param = 1 # Amount of Squeezing

# Optimal parameters for N = 21 based on the paper
phi_ans = [-0.353282, 0.902028, -0.518804, -0.447977, 0.447404, 0.231963, -0.307409, -0.191352, 0.354135, 0.141061, -0.544457, -0.310069, 0.685625, 0.623302, -0.144266, -0.808851, -0.102101, -0.682909, 1.02161, 0.534188, -0.54781]
phi = [np.pi/2 for i in range(N + 1)]

#%%
# Find all possible combinations of [+1, -1] for lists of length N.
# Number of combinations is given by 2^N.
possible_sj = [-1, +1]

full_list = [list(p) for p in product(possible_sj, repeat = N)]
for i in full_list:
    i.insert(0, -1) #s0 = -1 
    i.append(-1) #sN+1 = -1

#%%
def airy(t, n):
    """
    Approximation solution to space intgeral

    Error 1: Term needs to be multipled by 1/sqrt(pi) to be the approximate integration value
    """
    beta = t * n
    return 4.691 * np.exp(-beta/3) * special.airy(0.231 - 0.693 * beta)[0] / np.sqrt(np.pi)

def terms(j, sj_list, phi_list):
    """
    Generate each term within the product
    """
    power = (sj_list[j+1] - sj_list[j])/2

    numerator = np.exp(1j * phi_list[j]) + (-1)**(power**2) * np.exp(-1j * phi_list[j])
    denominator = 2 * 1j**power

    return numerator/denominator

def func(phi_list, full_list):
    components = []
    exponents = []

    for ls in full_list:
        inter = []
        #print('\n', ls, np.sum(ls[1:-1]))
        for i in range(len(ls) - 1):
            #print(terms(i, ls, phi_list))
            inter.append(terms(i, ls, phi_list))
        exponents.append(np.sum(ls[1:-1])+3.5)
        components.append(np.prod(inter))

    approx_integral = [airy(t, i) for i in exponents]
    #print('\n', components)
    #print(approx_integral)

    results = []
    for i in range(len(components)):
        results.append(components[i] * approx_integral[i])

    return 1.0 - np.absolute(np.sum(results))**2
# %%
#print(func(phi_ans, full_list))
#print(func([np.pi/4 for i in range(N+1)], full_list))

root = minimize(func, [np.random.uniform(-np.pi/2, np.pi/2) for i in range(N+1)], args = (full_list))
print(root.message)
print("Solution for Optimal Angles: ", root.x)
print("Calculated infidelity: ", np.round(func(root.x, full_list), 3))

# %%
# To Show Error 1 
""" import scipy.integrate as integrate

def exact(x, beta):
    return (1/np.sqrt(np.pi)) * np.exp(-x**2) * np.exp(-1j * x**3) * np.exp(1j * beta * x)

def approx(beta):
    return 4.691 * np.exp(-beta/3) * special.airy(0.231 - 0.693 * beta)[0] / np.sqrt(np.pi)

trial = [0.1*i for i in range(1,11)]

res_exact = []
res_approx = []
for k in trial:
    res_exact.append(integrate.quad(exact, -np.inf, np.inf, args = k)[0])
    res_approx.append(approx(k))

plt.plot(trial, res_exact, label = 'Numerical Integration')
plt.plot(trial, res_approx, label = 'Approximate Airy Function with 1/np.sqrt(pi)')
plt.legend() """
# %%
# Generate Fourier Coefficients
import scipy.integrate as integrate

def integrand(x, index, time):
    return np.exp(1j * 1 * x**3) * np.exp(-1j * index * time * x)

def coeff(time, index):
    norm = time / (2 * np.pi)
    res = np.real(integrate.quad(integrand, -np.pi/time, np.pi/time, args = (index, time))[0])
    return norm * res

time = [0.1 * i for i in range(1, 20)]

check = []
for i in range(100):
    check.append(coeff(0.42, i))

plt.plot(np.absolute(check), 'x-')
plt.hlines(y = np.absolute(check[10]), xmin = 0, xmax = 100, color = 'r')
plt.ylabel("Fourier Amplitude (Absolute Value)")
plt.xlabel("Fourier Index")
plt.title("Time = 0.42")

# %%
