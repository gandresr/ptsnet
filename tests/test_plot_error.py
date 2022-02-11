from ptsnet.utils.io import get_example_path
from ptsnet.utils.analytics import compute_simulation_times_per_step

compute_simulation_times_per_step(
    inpfile = get_example_path('BWSN_F'),
    time_steps = [0.0005, 0.001, 0.0015],
    steps = 100,
    count=3
)

# from ptsnet.graphics.static import plot_times_per_step
# plot_times_per_step()