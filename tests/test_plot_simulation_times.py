from ptsnet.utils.io import get_example_path
from ptsnet.utils.analytics import compute_simulation_times

compute_simulation_times(
    inpfile = get_example_path('BWSN_F'),
    time_steps = [0.0005],
    plot = False,
    steps = 10,
    count = 4,
    duration = 20,
    max_num_processors = 8,
    # environment = 'tacc',
    # allocation = 'LEAP-HI-2021'
)