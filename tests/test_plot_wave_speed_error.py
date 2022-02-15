from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path
from ptsnet.graphics.static import plot_wave_speed_error

sim = PTSNETSimulation(
    inpfile = get_example_path('BWSN_F'),
    settings = {
        'time_step' : 0.1,
        'duration' : 0.1*1000,
        'wave_speed_method': 'user'
    })

plot_wave_speed_error(sim, 'BWSNerror1.pdf')