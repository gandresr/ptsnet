from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path
from ptsnet.graphics.static import plot_wave_speed_error
import numpy as np

sim = PTSNETSimulation(
    inpfile = get_example_path('BWSN_F'),
    settings = {
        'time_step' : 0.1,
        'duration' : 0.1*1000,
        'wave_speed_method': 'user'
    })

print(sim)
covered = np.sum(np.abs(sim.ss['pipe'].desired_wave_speed/sim.ss['pipe'].wave_speed - 1) < 0.1)
print(covered/sim.wn.num_pipes)
plot_wave_speed_error(sim, 'BWSNerror1.pdf', intervals=[0,10,40,70,100])
