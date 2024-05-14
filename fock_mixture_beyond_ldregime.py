#%%
import numpy as np
import matplotlib.pyplot as plt
import math
from qutip import *

options = Options()
options.num_cpus = 5
options.nsteps = 100000

nMax = 50
ld_param = 0.5
rabi_freq = 1

a = tensor(qeye(2), destroy(nMax))
sM = tensor(destroy(2), qeye(nMax))

# %%
def term(n, j, kMin):
    factor = np.sqrt(n - j - 1)

    terms = []
    for k in range(kMin, j):
        terms.append(n-k)
    prod = np.prod(terms)

    denom = math.factorial(j) * math.factorial(j + 1)

    return factor * prod / denom
# %%
ls_freq = []
ls_freq2 = []
ans = []

ld_param = 0.1
rabi_freq = np.pi

for n in range(20):


    check = []
    for a in range(n):
        for b in range(n):
            first_term = term(n, a, kMin = 0)
            second_term = term(n, b, kMin = 1)

            check.append(ld_param**(a + b) * first_term * second_term)

    check2 = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                first_term = term(n, a, kMin = 0)
                second_term = term(n, b, kMin = 1)
                third_term = term(n, c , kMin = 0)

                check2.append(ld_param**(a + b + c) *np.sqrt(n+1) * first_term * second_term * third_term)


    freq = (rabi_freq/2) * ld_param * np.exp(-0.5 * ld_param**2) * np.sum(check)
    freq2 = (rabi_freq/2) * ld_param * np.exp(-0.5 * ld_param**2) * np.sum(check2)

    ls_freq.append(freq)
    ls_freq2.append(freq2)

    ratio = freq2 / freq
    mod_pi = np.mod(ratio, np.pi)

    ans.append(mod_pi)

plt.plot(ans, 'x')
for i in range(1, 4):
    plt.hlines(i, 0, 20, 'r')
plt.xticks([n for n in range(20)])

# %%
