import numpy as np
import matplotlib.pyplot as plt

from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path

inpfile = '/home/gandresr/Documents/GitHub/ptsnet/ptsnet/examples/TNET3_HAMMER.inp'

# These are time step values necessary to compite weak scaling
dt = [0.03955, 0.03955, 0.03955, 0.039648, 0.039748, 0.04, 0.04043, 0.0414, 0.04358, 0.051359]
dt = dt[::-1]
dt = [dt[i]/(2**i) for i in range(10)]

num_points_per_processor = []
num_processors = []

for i in range(10):
    sim = PTSNETSimulation(
        inpfile = inpfile,
        settings = {
            'duration': 20,
            'time_step': dt[i],
            'period' : 0,
            'wave_speed_method': 'user',
            'profiler_on' : True,
            'save_results' : True
        })
    num_points_per_processor.append(sim.num_points/(2**i))
    num_processors.append(2**i)
    sim.run()

print(num_processors)
print(num_points_per_processor)
plt.plot(num_processors, num_points_per_processor, 'x-')
plt.ylim(800, 1200)
plt.title("Number of Processors")
plt.title("Number of Tasks per Processor")
plt.savefig('scaling.png')