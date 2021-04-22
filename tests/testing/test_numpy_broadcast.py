#%%
import numpy as np

N = 30000000

x = np.arange(N)
y = np.random.randint(N//100, size=N//100)
z = np.zeros(N//100)

#%%
%%timeit
z[:] = x[y]
