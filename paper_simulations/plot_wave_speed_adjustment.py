import os
import numpy as np
import wntr
import matplotlib.pyplot as plt
import pickle

from phammer.simulation.sim import HammerSimulation
from phammer.utils.io import get_root_path

ROOT = get_root_path()
inpfile = os.path.join(ROOT, os.pardir, 'example_files', 'BWSN_F.inp')
PICKLED_FOLDER = 'pickled'
pfile = os.path.join(PICKLED_FOLDER,'phi_bwsn.dat')
os.makedirs(PICKLED_FOLDER, exist_ok=True)
wave_speed = 1000
N = 100

time_steps = np.linspace(5e-6, 0.0005791199999999999, N)
phi_norm = np.zeros(N)
num_points = np.zeros(N)

if not os.path.isfile(pfile):
    for i, t in enumerate(time_steps):
        sim = HammerSimulation(
            inpfile,
            {
                'time_step' : t,
                'duration' : t*2,
                'skip_compatibility_check' : False,
                'warnings_on' : False,
                'show_progress' : False,
            },
            period = 219,
            default_wave_speed = wave_speed)
        phi_norm[i] = np.linalg.norm(np.abs(sim.ic['pipe'].wave_speed/wave_speed - 1), 2)
        num_points[i] = sim.num_points
        print("%d %%" % ((i+1)*100/(N)))

    with open( pfile , 'wb') as f:
        pickle.dump([time_steps, phi_norm, num_points], f)
else:
    with open( pfile , 'rb') as f:
        time_steps, phi_norm, num_points = pickle.load(f)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time step (s)')
ax1.set_ylabel('$||\phi||^2$', color=color)
ax1.plot(time_steps, phi_norm, '.-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Number of points', color=color)  # we already handled the x-label with ax1
ax2.plot(time_steps, num_points, '.-', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Wave Speed Adjustment & Problem Size")

fig.tight_layout()  # otherwise the right y-label is slightly clipped


os.makedirs('figures', exist_ok=True)
plt.savefig(os.path.join('figures','ws_adjustment.pdf'))

# plt.show()