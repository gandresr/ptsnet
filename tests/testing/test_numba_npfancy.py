from numba import jit
import numpy as np
from time import time

@jit(nopython = True)
def operation(x, y, idx):
    x[idx] /= y[idx]

x = np.random.rand(1000000)
y = np.random.rand(1000000)

idx = np.random.randint(1000000, size = 20000)

t = time()
operation(x, y, idx)
print(time() - t)