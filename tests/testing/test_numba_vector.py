import numpy as np

from numba import jit
from time import time

N = 3000000
a = np.arange(1, N, dtype = np.float)
b = np.arange(1, N, dtype = np.float)

#selectors
x = np.arange(N)[0::357]

@jit(nopython = True)
def jit_divide(A, B, X):
    A[X] /= B[X]

def divide(A, B, X):
    A[X] /= B[X]

t = time()
jit_divide(a,b,x)
print(time() - t)

t = time()
divide(a,b,x)
print(time() - t)