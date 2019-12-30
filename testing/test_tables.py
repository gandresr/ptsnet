import numpy as np

from phammer.arrays import Table, Table2D

time_steps = 20
num_points = 10

properties = {
    'flow' : np.float,
    'head' : np.float
}

index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
T = 10
results = Table(properties, num_points, index)
pool = Table2D(properties, num_points, T, index)
results.flow['a']
x = pool[[1,4,5]]
x.flow['e']