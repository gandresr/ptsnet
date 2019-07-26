import numpy as np
import ctypes
from functools import reduce

x = [ctypes.c_float(i) for i in range(10)]
index = [1, 4, 7]

y = [ctypes.pointer(x[i]) for i in index]

def sy(y):
    s = 0
    for p in y:
        s += p[0]
    print(s)

sy(y)

x += 1
x -= 1
yy = [ctypes.pointer(ctypes.c_float(x[i])) for i in index]
sy(yy)