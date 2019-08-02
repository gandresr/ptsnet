import numpy as np

from phammer.arrays.tables import Table

time_steps = 20
num_points = 10

properties = {
    'flow' : np.float,
    'head' : np.float
}

index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
results = Table(properties, num_points, index)
results.flow['a']
