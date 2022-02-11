from multiprocessing.sharedctypes import Value
import numpy as np
import os, psutil

from ptsnet.utils.io import create_temp_folder, get_temp_folder
from ptsnet.simulation.constants import FILE_TEMPLATE
from ptsnet.simulation.sim import PTSNETSimulation

def compute_wave_speed_error(sim):
    ws = sim.ss['pipe'].wave_speed
    dws = sim.ss['pipe'].desired_wave_speed
    return np.abs(ws - dws)*100 / dws

def compute_num_processors(sim, plot=False, count=4, nprocessors=None):
    if type(sim) is not PTSNETSimulation: raise ValueError("'sim' must be a PTSNETSimulation")
    max_processors = psutil.cpu_count(logical=False) if nprocessors is None else nprocessors
    create_temp_folder()
    temp_file_path = os.path.join(get_temp_folder(), "temp_sim.py")
    compute_file_path = os.path.join(get_temp_folder(), "compute_processors_sim.py")

    # Modify settings for performance simulation only
    sim.settings.profiler_on = True
    sim.settings.duration = 1000*sim.time_step
    sim.settings.save_results = True

    # Write Python script to run simulations
    with open(temp_file_path, "w") as f:
        f.write(FILE_TEMPLATE.format(inpfile=sim.inpfile, settings=sim.settings.to_dict(simplified=True)))
        f.write("sim.run()")
    cwd = os.getcwd()
    processors = np.linspace(1, max_processors, count, dtype=int)

    # Write bash executable
    bash_path = os.path.join(cwd, "compute_num_processors.sh")
    with open(bash_path, "w") as f:
        i = 1
        f.write(f"echo 'Evaluating Performance (this might take a few minutes)'\n")
        for p in processors:
            f.write(f"mpiexec -n {p} python3 {temp_file_path} &> log.txt\n")
            f.write(f"echo '({int(100*i/len(processors))}%) Finished run {i}/{len(processors)}'\n")
            i += 1
        f.write(f"rm log.txt\n")
        f.write(f"python3 {compute_file_path}\n")

    # Write Python script to compute number of processors
    fcontent = "import matplotlib.pyplot as plt\n" if plot else ""
    fcontent += \
        "from pprint import pprint\n" + \
        "import numpy as np\n" + \
        "from tqdm import tqdm\n" + \
        "from ptsnet.simulation.sim import PTSNETSimulation\n" + \
        "from ptsnet.results.workspaces import num_workspaces\n" + \
        "from kneed import KneeLocator\n" + \
        f"workspaces = [num_workspaces()-1-i for i in range({count})]\n" + \
        "times = {}\n" + \
        "for w in workspaces:\n" + \
        "    with PTSNETSimulation(w) as sim:\n" + \
        "        exchange_data_time = np.mean(sim.profiler.summary['comm_jobs'].exchange_data)\n" + \
        "        barrier1_time = np.mean(sim.profiler.summary['comm_jobs'].barrier1)\n" + \
        "        barrier2_time = np.mean(sim.profiler.summary['comm_jobs'].barrier2)\n" + \
        "        run_step_time = np.mean(sim.profiler.summary['step_jobs'].run_step)\n" + \
        "        total_time = exchange_data_time + barrier1_time + barrier2_time + run_step_time\n" + \
        "        times[int(sim.settings.num_processors)] = float(total_time)\n" + \
        "x = list(times.keys()); x.sort()\n" + \
        "y = [times[i] for i in x]\n" + \
        "optimal = -1; kl = None\n" + \
        "if y[1] > y[0]:\n" + \
        "    optimal = 1\n" + \
        "else:\n" + \
        "    try:\n" + \
        "        kl = KneeLocator(x, y, curve='convex', direction='decreasing')\n" + \
        "        optimal = kl.knee\n" + \
        "    except:\n" + \
        "        pass\n" + \
        "    if optimal == -1: optimal = x[-1]\n" + \
        "print(f'\\nProcessor Times: \\n')\n" + \
        "pprint(times)\n" + \
        "print(f'\\n--> Recommended number of processors: {optimal}\\n')\n"

    if plot:
        fcontent += "if kl: kl.plot_knee(); plt.show()\n"

    with open(compute_file_path, "w") as f:
        f.write(fcontent)

    print("\nExecute the following command on your terminal:")
    print(f"bash {bash_path}\n")
    exit()