import numpy as np
from time import time
from numba import njit
import matplotlib.pyplot as plt

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
T = int(2)
HH = [np.ones(N).astype('float64')*1e-6]
for i in range(T-1):
    HH.append(np.zeros(N).astype('float64'))
QQ = [np.ones(N).astype('float64')*1e-6]
for i in range(T-1):
    QQ.append(np.zeros(N).astype('float64'))
B = np.ones(N).astype('float64')
R = np.ones(N).astype('float64')

@njit(parallel=True)
def calc(H1, H2, Q1, Q2):
    for i in range(1,N-1):
        H2[i] = (H1[i-1] + B[i]*Q1[i-1]) * (B[i] + R[i]*abs(Q1[i+1])) + (H1[i-1] - B[i]*Q1[i+1]) * (B[i] + R[i]*abs(Q1[i-1]))
        Q2[i] = (H1[i-1] - B[i]*Q1[i+1]) * (B[i] + R[i]*abs(Q1[i-1]))

clk = Clock()

calc(HH[0], HH[1], QQ[0], QQ[1])

tt = 0
clk.tic()
for t in range(1,T-1):
    calc(HH[t], HH[t+1], QQ[t], QQ[t+1])
clk.toc()

print(HH[T-1])
plt.plot(HH[T-1])
plt.show()

