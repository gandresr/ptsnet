import numpy as np

from phammer.arrays.tables import Table2D

time_steps = 20
num_points = 10

properties = {
    'flow' : np.float,
    'head' : np.float
}

results = Table2D(properties, time_steps, num_points)

