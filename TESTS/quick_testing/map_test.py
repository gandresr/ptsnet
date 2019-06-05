import numpy as np
from time import time
class Clock:
    def __init__(self):
        self.clk = time()

    def tic(self):
        self.clk = time()

    def toc(self):
        print('Elapsed time: %f seconds' % (time() - self.clk))

def fcn(a, i):
    a[i] += 2

x = [1 for i in range(1000000)]
y = [1 for i in range(1000000)]

clk = Clock()

clk.tic()
for i in range(len(x)):
    fcn(x,i)
clk.toc()

def fcn2(a):
    return a + 2

clk.tic()
y = [*map(fcn2, y)]
clk.toc()

clk.tic()
z = np.ones(1000000)
z += 2
clk.toc()

assert x == y
assert y == list(z)