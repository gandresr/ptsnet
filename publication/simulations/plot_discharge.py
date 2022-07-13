from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 32
sim = PTSNETSimulation(
    inpfile = get_example_path('TNET3'),
    settings = {
        'duration': 20,
        'time_step': 0.005,
        'period' : 0,
        'default_wave_speed' : 1200})

sim.add_burst('JUNCTION-73', 0.02, 1, 2)
sim.run()
fig = plt.figure(figsize=(14.5, 8))
fig.clf()
plt.plot(sim['time'], sim['node'].leak_flow['JUNCTION-73'], linewidth=4, label='73', color='#007aff')
plt.legend()
plt.xlim(0,20)
plt.ylabel('Flowrate [$m^3/s$]', fontsize=28)
plt.xlabel('Time [s]', fontsize=28)
plt.tight_layout()
plt.savefig('discharge.pdf')