#%%
import numpy as np
import scipy.integrate as integrate
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import sympy as sym

gate_time = 500

""" # Maximum output frequency for AWG = 200 MHz
# This limits the maximum frequency Fourier component that can be used for the pulse generation
max_n = int(np.floor(200 * 10**(6) * gate_time))
print("Maximum frequency component =", max_n) """

# ------------- Calculate Mode Frequencies
w_carr = 112.943265 
w_Ax = 113.347860 
w_R1 = 114.038841 
w_R2 = 114.314120 

w_Ax_IP = w_Ax - w_carr
w_Ax_OP1 = np.sqrt(3) * w_Ax_IP
w_Ax_OP2 = np.sqrt(29/5) * w_Ax_IP
w_R1_IP = w_R1 - w_carr
w_R1_OP1 = np.sqrt(w_R1_IP**2 - w_Ax_IP**2)
w_R1_OP2 = np.sqrt(w_R1_IP**2 - 12/5 * w_Ax_IP**2)
w_R2_IP = w_R2 - w_carr
w_R2_OP1 = np.sqrt(w_R2_IP**2 - w_Ax_IP**2)
w_R2_OP2 = np.sqrt(w_R2_IP**2 - 12/5 * w_Ax_IP**2)

# w = [w_Ax_IP, w_Ax_OP1, w_Ax_OP2, w_R1_IP, w_R1_OP1, w_R1_OP2, w_R2_IP, w_R2_OP1, w_R2_OP2]
# w = [w_Ax_IP, w_R1_IP, w_R2_IP]

w = [2 * np.pi * w_R2_IP, 2 * np.pi * w_R2_OP1, 2 * np.pi * w_R2_OP2]

LD_param = 0.1

ld = [[1/np.sqrt(3) * LD_param, 1/np.sqrt(3) * LD_param, 1/np.sqrt(3) * LD_param], 
      [- 1/np.sqrt(2) * LD_param, 0, 1/np.sqrt(2) * LD_param], 
      [1/np.sqrt(6) * LD_param, -2 * 1/np.sqrt(6) * LD_param, 1/np.sqrt(6) * LD_param]]

#%%
# ------------- Calculate Null Vectors
p = len(w)
nmin = int(np.round(gate_time * w_R2_OP2)) - 10
nmax = int(np.round(gate_time * w_R2_IP)) + 10
print("nMin", nmin, "nMax", nmax)
NA = nmax - nmin

M = np.empty([p, NA])
M_err = np.empty([p, NA])

def integrand(t, p, n):
    return np.sin(2 * np.pi * (n + 1) * t / gate_time) * np.sin(w[p] * ((gate_time/2) - t))

for i in range(p):
    for j in range(NA):
        M[i, j], M_err[i, j] = integrate.quad_vec(integrand, 0, gate_time, args = (i, j), epsabs = 1.49e-8)

Gamma = np.matmul(M.transpose(), M)

#%%
M_null = np.linalg.eigh(Gamma)

plt.figure()
plt.yscale('log')
plt.scatter([i for i in range(len(M_null[0]))], np.absolute(M_null[0]))

#Based on plot, the last three eigenvalues are considered > 0
#Extract the eigenvectors with these non-zero eigenvalues

null_vectors = [[] for i in range(NA - p)]

for i in range(NA-p):
    null_vectors[i] = M_null[1][i].transpose()
    
plt.figure()
plt.plot(null_vectors)

# ------------- Calculate
#%%
D = np.empty([NA, NA])

t2, t1 = sym.symbols('t2 t1')

def f(*numbers):
    n = numbers[0]
    m = numbers[1]
    p = numbers[2]
    return sym.sin(2 * np.pi * (n + 1) * t2 / gate_time) * sym.sin(w[p] * (t2 - t1)) * sym.sin(2 * np.pi * (m + 1) * t1 / gate_time)

for i in range(NA):
    if i % ((nmax - nmin)/10) == 0:
        print( (i * 100 / NA), '% done')
    for j in range(NA): 
        store = []
        for k in range(p):
            elem = sym.integrate(f(i, j, k), (t1, 0, t2), (t2, 0, gate_time))
            store.append(elem)
        D[i, j]= np.sum(store)

S = 0.5 * np.add(D, D.transpose())
print(S)

#%%
N0 = (NA) - p

R = np.empty([N0, N0])

for i in range(N0):
    for j in range(N0):
        inter = S @ null_vectors[j]
        hi = null_vectors[i].transpose()
        R[i,j] = hi @ inter

#print(R)

eigval, eigvec= linalg.eig(R)
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
t = np.arange(0, gate_time, 1.0 * 10**(-6))

def g(t, Vector):
    return np.sum([Vector[i] * np.sin (2 * np.pi * (i + 1) * t / gate_time) for i in range(0, nmax - nmin)])

ans = [g(time, Vector2) for time in t]

plt.plot(t, ans)

plt.figure()
#plt.yscale('log')
plt.plot([i/gate_time for i in range(nmin, nmax)], np.absolute(Vector2))
plt.vlines([i / (2*np.pi) for i in w], ymin = 10**(-6), ymax = 0.3)
# %%
