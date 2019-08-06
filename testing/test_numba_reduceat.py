#%%
import numpy as np
from numba import vectorize, jit, prange
from time import time

N = 300000000
a = np.arange(N)
b = np.arange(N)[0::4]

def radd(A, B):
    return np.add.reduceat(A, B)

@jit(nopython = True, nogil=True, parallel=True)
def jit_radd(A, B):
    x = np.zeros(len(B))
    for i in prange(len(B)-1):
        x[i] = np.sum(A[B[i]:B[i+1]])
    x[-1] = np.sum(A[B[-1]:])
    return x

y = jit_radd(a, b)


#%%
%%timeit
y = jit_radd(a, b)
#%%
# %%
%%timeit
x = radd(a, b)

# Conclusion:
#  np.ufunc.reduceat performs as a numba jitted function
#  However, if the number of elements in the array is large,
#  around 1M entries, it is better to use the jitted func
#  in parallel with nogil.