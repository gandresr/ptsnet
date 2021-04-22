from time import time
import numpy as np

N = 100000
x = np.zeros(N)
xx = np.zeros(N)
y = np.arange(N)

t = time()
for i in range(len(y)):
    if y[i] >= N/10:
        x[i] = y[i] + 1
    elif y[i] >= N/100:
        x[i] = y[i] + 2
    elif y[i] >= N/1000:
        x[i] = y[i] + 3
    elif y[i] >= N/2000:
        x[i] = y[i] + 5
t1 = (time() - t)

t = time()
xx[y >= N/2000] = y[y >= N/2000] + 5
xx[y >= N/1000] = y[y >= N/1000] + 3
xx[y >= N/100] = y[y >= N/100] + 2
xx[y >= N/10] = y[y >= N/10] + 1
t2 = (time() - t)

print("numpy speedup %f" % (t1/t2))
x[x != xx]
