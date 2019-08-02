import numpy as np

from phammer.arrays.tables import Table2D

time_steps = 20
num_points = 10

properties = {
    'flow' : np.float,
    'head' : np.float
}

index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
results = Table2D(properties, num_points, time_steps, index)
results.flow[0:1]
