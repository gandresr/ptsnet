
import glob, os
import numpy as np
import ntpath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
times_processor = {}

x = np.array([i*1000 for i in range(1,10)])
y = np.array([i*1000 for i in range(1,5)])

X = np.zeros((9,4), dtype = int)
Y = np.zeros((9,4), dtype = int)

for j in range(4):
    for i in range(9):
        X[i, j] = x[i]
        Y[i, j] = y[j]
plt.ylabel("n")
plt.xlabel("m")

for fname in glob.glob("*.o*"):
    times = np.zeros((len(x),len(y)))
    with open(fname, 'r') as f:
        i = 0
        j = 0
        for line in f:
            if "SEQ: " in line:
                val = float(line[line.find(":") + 1 : line.find("sec")])
                times[i,j] = val
                print(i,j)
                i += 1
                if i % 9 == 0:
                    i = 0
                    j += 1
    bname = ntpath.basename(fname)
    k = int(bname[1:bname.find('.')])
    times_processor[k] = times
    print(times)

for k in range(1,25):
    ss = np.nan_to_num( times_processor[1] / (k*times_processor[k]))
    ax.plot_surface(X, Y, ss, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.title("Strong Scaling Efficiency")
plt.show()