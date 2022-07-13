import os
import numpy as np

import matplotlib.pyplot as plt
from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path
from time import time

t1 = time()
sim = PTSNETSimulation(
    inpfile = get_example_path('TNET3'),
    settings = {
        'time_step' : 0.1,
        'save_results' : False,
        'default_wave_speed' : 1000,
        'wave_speed_method' : 'user'
})

# sim.add_curve('V_BUTTERFLY', 'valve',
#     [1, 0.8, 0.6, 0.4, 0.2, 0],
#     [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])
# sim.assign_curve_to('V_BUTTERFLY', sim.all_valves)

sim.run()
print(f'Elapsed time: {time()-t1} s')

# plt.plot(sim['time'], sim['node'].head['JUNCTION-23'])
# plt.savefig('test.png')
exit()
assert np.allclose(sim.ss['valve'].flowrate, np.array([0.028]), atol=1e-3)

assert np.allclose(sim.ss['valve'].head_loss, np.array([1.388]), atol=1e-3)

assert np.allclose(sim.ss['valve'].K, np.array([4.885e-05]))

assert np.allclose(sim.ss['pipe'].flowrate,
    np.array([
        0.286,
        0.135,
        0.151,
        0.118,
        0.016,
        0.065,
        0.028,
        0.037,
        0.102]), atol =  1e-3)

assert np.allclose(sim.ss['pipe'].head_loss,
    np.array([
        1.662e-07,
        7.835e-08,
        8.783e-08,
        6.887e-08,
        9.481e-09,
        3.792e-08,
        1.646e-08,
        2.147e-08,
        5.939e-08]))

test_initial_conditions()