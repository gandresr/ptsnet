import numpy as np
from time import time
from numba import njit

class Clock:
    """Wall-clock time
    """
    def __init__(self):
        self.clk = time()

    def tic(self):
        """Starts timer
        """
        self.clk = time()

    def toc(self):
        """Ends timer and prints time elapsed
        """
        print('Elapsed time: %f seconds' % (time() - self.clk))

N = int(5000)
H = np.random.rand(N).astype('float64')
H2 = np.random.rand(N).astype('float64')
Q = np.random.rand(N).astype('float64')
B = np.random.rand(N).astype('float64')
R = np.random.rand(N).astype('float64')

@njit(nopython=True, parallel=True)
def calc(H2):
    for i in range(1,len(H)-1):
        H2[i] = (H[i-1] + B[i]*Q[i-1]) * (B[i] + R[i]*abs(Q[i+1])) + (H[i-1] - B[i]*Q[i+1]) * (B[i] + R[i]*abs(Q[i-1]))

clk = Clock()

calc(H2)

clk.tic()
for i in range(4000):
    calc(H2)
clk.toc()

