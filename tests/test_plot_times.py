from ptsnet.utils.io import get_example_path
from ptsnet.utils.analytics import compute_simulation_times_per_step

compute_simulation_times_per_step(
    inpfile = get_example_path('TNET3_HAMMER'),
    time_steps = [0.0015, 0.001],
    plot = True,
    steps = 2500,
    count = 4,
    duration = 5,
)