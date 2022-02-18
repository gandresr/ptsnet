from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path
from ptsnet.utils.analytics import compute_num_processors

sim = PTSNETSimulation(
    inpfile = get_example_path('BWSN_F'),
    settings = {
        'time_step' : 0.0005,
        'save_results' : False,
        'default_wave_speed' : 1000,
        'wave_speed_method' : 'user'
})

compute_num_processors(sim, plot=False, count=4, steps=2500, max_num_processors=8, environment='tacc', allocation='LEAP-HI-2021')