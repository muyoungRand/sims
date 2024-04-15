#%%
import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import sympy as sym
import sys

np.set_printoptions(threshold=sys.maxsize)
#%%
gate_time = 300 * 10**(-6)

# ------------- Calculate Mode Frequencies
w_carr = 112.943265 * 10**(6)
w_Ax = 113.347860 * 10**(6)
w_R1 = 114.038841 * 10**(6)
w_R2 = 114.314120 * 10**(6)

w_Ax_IP = w_Ax - w_carr
w_Ax_OP1 = np.sqrt(3) * w_Ax_IP
w_Ax_OP2 = np.sqrt(29/5) * w_Ax_IP
w_R1_IP = w_R1 - w_carr
w_R1_OP1 = np.sqrt(w_R1_IP**2 - w_Ax_IP**2)
w_R1_OP2 = np.sqrt(w_R1_IP**2 - 12/5 * w_Ax_IP**2)
w_R2_IP = w_R2 - w_carr
w_R2_OP1 = np.sqrt(w_R2_IP**2 - w_Ax_IP**2)
w_R2_OP2 = np.sqrt(w_R2_IP**2 - 12/5 * w_Ax_IP**2)

mode_freq = [2 * np.pi * w_R2_IP, 2 * np.pi * w_R2_OP1, 2 * np.pi * w_R2_OP2]

# ------------- Generate ion-specific LD parameter values
LD_param = 0.1

ld = [[1/np.sqrt(3) * LD_param, 1/np.sqrt(3) * LD_param, 1/np.sqrt(3) * LD_param], 
      [- 1/np.sqrt(2) * LD_param, 0, 1/np.sqrt(2) * LD_param], 
      [1/np.sqrt(6) * LD_param, -2 * 1/np.sqrt(6) * LD_param, 1/np.sqrt(6) * LD_param]]

#%%
gate_time = 300

mode_freq = [2 * np.pi * 2.26870,
             2 * np.pi * 2.33944,
             2 * np.pi * 2.39955,
             2 * np.pi * 2.44820,
             2 * np.pi * 2.48038]

ld = [[0.01248, -0.05479, 0.08428, -0.05440, 0.01243],
      [0.03473, -0.07263, -0.00002, 0.07306, -0.03514],
      [0.06091, -0.03150, -0.05848, -0.03098, 0.06094],
      [0.07149, 0.03406, -0.00021, -0.03459, -0.07163],
      [-0.04996, -0.05016, -0.05013, -0.04991, -0.04946]]

#%%
# ------------- Calculate Null Vectors
nmin = 0
nmax = 1000

p = len(mode_freq)
NA = nmax - nmin

# Setup M matrix
M = np.empty([p, NA])
M_err = np.empty([p, NA])

# Symbolic Integration
t, T, w = sym.symbols('t T w', positive = True)
n = sym.symbols('n', integer = True, positive = True)

integrand = sym.sin(2 * sym.pi * n * t / T) * sym.sin(w * ( (T/2) - t))
res = sym.integrate(integrand, (t, 0, T))

# Evaluiate elements of M matrix
for i in range(p):
    for j in range(NA):
        M[i, j] = res.subs([(T, gate_time), (w, mode_freq[i]), (n, j+1)])
        
# Find eigenvalues and null vectors
Gamma = np.matmul(M.transpose(), M)

M_null = np.linalg.eigh(Gamma)

plt.figure()
plt.yscale('log')
plt.scatter([i for i in range(len(M_null[0]))], np.absolute(M_null[0]))

# Last p eigenvalues are non-zero (numpy arranges them in ascending order)
# So only the first (NA - p) eigenvalues are the null vectors
null_vectors = [[] for i in range(NA - p)]

for i in range(NA - p):
    null_vectors[i] = M_null[1][i]
    
plt.figure()
plt.plot(null_vectors)
# %%
# ------------- Calculate D Matrix
D = np.empty([NA, NA])

# Symbolic Integration
t2, t1, T, w = sym.symbols('t2 t1 T w', positive = True)
n, m = sym.symbols('n m', integer = True, positive = True)

integrand1 = sym.sin(2 * sym.pi * n * t2 / T) * sym.sin(w * (t2 - t1)) * sym.sin(2 * sym.pi * m * t1 / T)

res1 = sym.integrate(integrand1, (t1, 0, t2))
res1 = sym.simplify(res1)

res2 = sym.integrate(res1, (t2, 0, T))
res2 = sym.simplify(res2)

#%%
ion1 = 1
ion2 = 4

# Substitute Values
for i in range(NA):
    for j in range(NA):
        mode_specific_val = []
        for k in range(p):
            elem = ld[k][ion1] * ld[k][ion2] * res2.subs([(T, gate_time), (w, mode_freq[k]), (n, i+1), (m, j+1)])
            mode_specific_val.append(elem)
        D[i, j] = sum(mode_specific_val)
        
S = 0.5 * np.add(D, D.transpose())
print((S==S.T).all())
# %%
N0 = len(null_vectors)

R = np.empty([N0, N0])

for i in range(N0):
    for j in range(N0):
        inter = np.matmul(S, null_vectors[j])
        hi = np.matmul(null_vectors[i], inter)
        R[i,j] = hi
#%%
eigval, eigvec = np.linalg.eigh(R)
eigval = [np.absolute(i) for i in eigval]

max_eigval_index = np.where(eigval == max(eigval))[0][0]
max_eigval = eigval[max_eigval_index]
max_eigvec = eigvec[max_eigval_index]

v_k_max = (np.pi / (8 * max_eigval)) ** (1/2)
#v_k_max = 1

Lambda = v_k_max * max_eigvec
#print(Lambda)

Vector = np.array([Lambda[i] * null_vectors[i] for i in range(N0)])
#Vector2 = [Vector[0][i] + Vector[1][i] for i in range(0, nmax - nmin)]
Vector2 = np.add.reduce(Vector)
# %%
t = np.arange(0, gate_time, 1.0)

def g(t, Vector):
    return np.sum([Vector[i] * np.sin (2 * np.pi * (i + 1) * t / gate_time) for i in range(0, nmax - nmin)])

ans = [g(time, Vector2) for time in t]

#plt.plot(t, ans)

plt.figure()
plt.yscale('log')
plt.plot([i/gate_time for i in range(500, nmax)], np.absolute(Vector2[500:]))
#plt.vlines([i / (2*np.pi) for i in mode_freq], ymin = 10**(-6), ymax = 0.3)
# %%
