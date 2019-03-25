# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from math import pi
from datetime import datetime

# Parameters - Inputs for NN model

D = 0.5 # [m]
A = pi*D**2/4 # [m^2]

H0 = 1000 # [m]
Q0 = 1.06
V0 = Q0 / A # [m/s]

n = 4 # The pipe is divided in N parts
a = 2500 # [m/s]
L = 1000 # [m]
f = 0.02
tc = 1
s = [] # Valve setting

# Constants
t0 = 0 # [s] - This is not used, for every simulation, it is assumed t0 = 0
dt = 1.*L/(a*n)
tf = 1 # [s]
g = 32.2 # [m/s]
Z = lambda i: Z0*(1 - i/n)

# Variables
s = 1 - np.array([i*dt/tc for i in range(int(tf/dt))])
s[s<0] = 0
V = np.zeros((n+1,int(tf/dt)))
H = np.zeros((n+1,int(tf/dt)))

# Valve Openning
plt.plot(s)
plt.show()
print(tf/dt)

# Initial conditions are defined
for i in range(n+1):
    H[i,0] = H0 - i*(1.*L/n)*f*V0**2/(2*g*D)
    V[i,0] = V0

print(H[-1,0])
tt = []
startttime = datetime.now()
for j in range(1,int(tf/dt)):
    t = j*dt
    tt.append(t)
    for i in range(n+1):
        # Pipe start
        if i == 0:
            H[i,j] = H0
            V[i,j] = V[i+1, j-1] + g/a*(H0 - H[i+1, j-1]) - f*dt*V[i+1, j-1]*abs(V[i+1, j-1])/(2*D)
        # Pipe end  
        if i == n:
            V[i,j] = V0 * s[j]
            H[i,j] = H[i-1, j-1] - a/g*(V[i,j] - V[i-1,j-1]) - a/g*(f*dt*V[i-1,j-1]*abs(V[i-1,j-1])/(2*g))
        # Interior points
        if (i > 0) and (i < n):
            V1 = V[i-1,j-1]; H1 = H[i-1,j-1]
            V2 = V[i+1,j-1]; H2 = H[i+1,j-1]
            V[i,j] = 1/2*(V1 + V2 + g/a*(H1 - H2) - f*dt/(2*D)*(V1*abs(V1) + V2*abs(V2)))
            H[i,j] = 1/2*(a/g*(V1 - V2) + H1 + H2 - a/g*(f*dt/(2*D)*(V1*abs(V1) - V2*abs(V2))))
simtime = datetime.now() - startttime
print(simtime)

plt.plot(H[:,:].T)
plt.show()