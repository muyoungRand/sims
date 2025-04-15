#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy

# %%
# ---------- Symbolic Integration ---------- #
# M Matrix
t, T, w = sym.symbols('t T w', positive = True)
n = sym.symbols('n', positive = True, integer = True)

integrand1 = sym.sin(2 * sym.pi * n * t / T) * sym.sin(w * ((T/2) - t))
res1 = sym.simplify(sym.integrate(integrand1, (t, 0, T)).args[0][0])

# D Matrix
t2, t1, T, w = sym.symbols('t2 t1 T w', positive = True)
n, m = sym.symbols('n m', positive = True, integer = True)

integrand2 = sym.sin(2 * sym.pi * n * t2 / T) * sym.sin(w * (t2 - t1)) * sym.sin(2 * sym.pi * m * t1 / T)
res2_a = sym.simplify(sym.integrate(integrand2, (t1, 0, t2))).args[0][0]
res2 = sym.simplify(sym.integrate(res2_a, (t2, 0, T)))

res2_mn = res2.args[1]
res2_others = res2.args[3]

print(res1)
print(res2_mn)
print(res2_others)
# %%
# ---------- Relevant Numbers ---------- #
""" mode_freq = [2.26870, 2.33944, 2.39955, 2.44820, 2.48038]

freq = []
for i in mode_freq:
    freq.append(i * 2 * np.pi)

ld_param = [[0.01248, -0.05479, 0.08428, -0.05440, 0.01243],
            [0.03474, -0.07263, -0.00002, 0.07306, -0.03514],
            [0.06091, -0.03150, -0.05848, -0.03098, 0.06094],
            [0.07194, 0.03406, -0.00021, -0.03459, -0.07163],
            [-0.04996, -0.05016, -0.05013, -0.04991, -0.04946]
            ]
 """
mode_freq = [1.010, 1.201]

freq = []
for i in mode_freq:
    freq.append(i * 2 * np.pi)

ld_param = [[0.05, 0.05],
            [-0.029, 0.029]
            ]

ion_choice = [0, 1]

NA = [i for i in range(0, 600)]
tt = 300

#%%
M = np.empty([len(freq), len(NA)])

for i in range(len(freq)):
    for j in range(len(NA)):
        M[i, j] = res1.subs([(T, tt), (w, freq[i]), (n, NA[j])])

gamma = np.dot(M.T, M)
plt.matshow(M)
plt.colorbar()

""" eigval, eigvec = np.linalg.eig(gamma)
null_vectors2 = np.real(eigvec[len(freq):]) """

null_vectors = scipy.linalg.null_space(M)

# %%
D = np.empty([len(NA), len(NA)])

for i in range(len(NA)):
    for j in range(len(NA)): 
        ls = []
        if i == j:
            for k in range(len(freq)):
                val = res2_mn.subs([(T, tt), (w, freq[k]), (n, NA[i]), (m, NA[j])])
                ls.append(ld_param[k][ion_choice[0]] * ld_param[k][ion_choice[1]] * val[0])
        else:
            for k in range(len(freq)):
                val = res2_others.subs([(T, tt), (w, freq[k]), (n, NA[i]), (m, NA[j])])
                ls.append(ld_param[k][ion_choice[0]] * ld_param[k][ion_choice[1]] * val[0])
        D[i, j] = np.sum(ls)

S = 1/2 * (D + D.T)
plt.matshow(S)
plt.colorbar()
#%%
N0 = len(NA) - len(freq)

""" R = np.empty([N0, N0])

for i in range(N0):
    for j in range(N0):
        inter = np.dot(S, null_vectors.T[i])
        R[i, j] = np.dot(null_vectors.T[j].T, inter) """

R = np.dot(null_vectors.T, np.dot(S, null_vectors))
#R = (1/2) * (R + R.T)

plt.matshow(R)
plt.colorbar()
# %%
R_eigval, R_eigvec = np.linalg.eig(R)

k_max = np.where(np.absolute(R_eigval) == max(np.absolute(R_eigval)))[0][0]

v_k_max = np.sqrt(np.pi / (8 * np.absolute(R_eigval[k_max])))
Lambda = v_k_max * R_eigvec[:,k_max]
plt.plot(Lambda)

#%%
A_inter = [Lambda[i] * null_vectors[:,i] for i in range(N0)]
A = np.add.reduce(A_inter)

plt.yscale('log')
plt.plot(NA, np.absolute(A))

x = [int(np.round(i * tt)) for i in mode_freq]
y = [np.absolute(A[i- NA[0]]) for i in x]

plt.vlines(x, 0, y, 'r')

# %%
