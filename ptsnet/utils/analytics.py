from multiprocessing.sharedctypes import Value
from time import time
import numpy as np
import os, psutil, pickle

from ptsnet.utils.io import create_temp_folder, get_temp_folder
from ptsnet.simulation.constants import FILE_TEMPLATE
from ptsnet.simulation.sim import PTSNETSimulation, PTSNETSettings
from ptsnet.results.workspaces import num_workspaces

def compute_wave_speed_error(sim):
    ws = sim.ss['pipe'].wave_speed
    dws = sim.ss['pipe'].desired_wave_speed
    return np.abs(ws - dws)*100 / dws

def compute_num_processors(sim, plot=False, count=4, nprocessors=None, steps=2500):
    if type(sim) is not PTSNETSimulation: raise ValueError("'sim' must be a PTSNETSimulation")
    max_processors = psutil.cpu_count(logical=False) if nprocessors is None else nprocessors
    create_temp_folder()
    temp_file_path = os.path.join(get_temp_folder(), "temp_sim.py")
    compute_file_path = os.path.join(get_temp_folder(), "compute_processors_sim.py")

    # Modify settings for performance simulation only
    sim.settings.profiler_on = True
    sim.settings.duration = steps*sim.time_step
    sim.settings.save_results = True

    # Write Python script to run simulations
    with open(temp_file_path, "w") as f:
        f.write(FILE_TEMPLATE.format(workspace_id=None, inpfile=sim.inpfile, settings=sim.settings.to_dict(simplified=True)))
        f.write("sim.run()")
    cwd = os.getcwd()
    processors = np.linspace(1, max_processors, count, dtype=int)

    # Write bash executable
    bash_path = os.path.join(get_temp_folder(), "compute_num_processors.sh")
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
        "import numpy as np\n" + \
        "from pprint import pprint\n" + \
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
        fcontent += \
        "plt.plot(x, y, '-o')\n" + \
        "plt.axvline(x = optimal)\n" + \
        "plt.xlabel('Number of processors')\n" + \
        "plt.ylabel('Time [s]')\n" + \
        "plt.title('Average Time per Step')\n" + \
        "plt.savefig('knee.pdf')"

    with open(compute_file_path, "w") as f:
        f.write(fcontent)

    print("\nExecute the following command on your terminal:")
    print(f"bash {bash_path}\n")
    exit()

def compute_simulation_times_per_step(inpfile, time_steps, duration=20, plot=False, count=4, nprocessors=None, steps=2500):
    create_temp_folder()
    max_processors = psutil.cpu_count(logical=False) if nprocessors is None else nprocessors
    processors = np.linspace(1, max_processors, count, dtype=int)
    compute_file_path = os.path.join(get_temp_folder(), "compute_times_per_step.py")

    ii = 0
    for ts in time_steps:
        temp_file_path = os.path.join(get_temp_folder(), f"temp_sim_{ii}.py")
        # Write Python script to run simulations
        settings = PTSNETSettings(
            time_step = ts,
            duration = steps*ts,
            warnings_on = False,
            skip_compatibility_check = True,
            save_results = True,
            profiler_on = True,
            wave_speed_method = 'user'
        )
        with open(temp_file_path, "w") as f:
            f.write(FILE_TEMPLATE.format(workspace_id=None, inpfile=inpfile, settings=settings.to_dict(simplified=True)))
            f.write("sim.run()")
        cwd = os.getcwd()
        ii += 1

    sims = {(processors[ii], time_steps[jj]) : (ii, jj)  for ii in range(len(processors)) for jj in range(len(time_steps))}

    # Write bash executable
    bash_path = os.path.join(get_temp_folder(), "compute_times_per_step.sh")
    with open(bash_path, "w") as f:
        jj = 1
        f.write(f"echo 'Executing Simulations to Evaluate Performance'\n")
        f.write(f"echo '[This might take a few minutes]'\n")
        for ii, (p, ts) in enumerate(sims):
            temp_file_path = os.path.join(get_temp_folder(), f"temp_sim_{ii%len(time_steps)}.py")
            f.write(f"mpiexec -n {p} python3 {temp_file_path} &> log.txt\n")
            f.write(f"echo '({int(100*(1+ii)/len(sims))}%) Finished Simulation {1+ii}/{len(sims)} -> {p} processor(s) | time_step = {ts} s'\n")
            jj += 1
        f.write(f"rm log.txt\n")
        f.write(f"python3 {compute_file_path}\n")

    # Write Python script to compute simulation times
    export_path = os.path.join(get_temp_folder(), "exported_times.pkl")
    fcontent = \
    "import numpy as np\n" + \
    "import pickle\n" + \
    "from ptsnet.simulation.sim import PTSNETSimulation\n" + \
    "from ptsnet.results.workspaces import num_workspaces\n" + \
    f"sims = {sims}\n" + \
    f"processors = {str(list(processors))}\n" + \
    f"ntsteps = {len(time_steps)}; nproc = {count}\n" + \
    "init_times = np.zeros((ntsteps, nproc,))\n" + \
    "interior_times = np.zeros((ntsteps, nproc,))\n" + \
    "boundary_times = np.zeros((ntsteps, nproc,))\n" + \
    "comm_times = np.zeros((ntsteps, nproc,))\n" + \
    "totals = np.zeros((ntsteps, nproc,))\n" + \
    f"time_steps = {str(list(time_steps))}\n" + \
    "for ii, (p, ts) in enumerate(sims):\n" + \
    f"    jj = time_steps.index(ts) # index for time step\n" + \
    "    kk = processors.index(p)\n" + \
    "    workspace_id = num_workspaces() - len(sims) + ii\n" + \
    "    with PTSNETSimulation(workspace_id) as sim:\n" + \
    "        init_times[jj, kk] = \\\n" + \
    "            (sim.profiler.summary['get_partition'] + \\\n" + \
    "            sim.profiler.summary['_create_selectors'] + \\\n" + \
    "            sim.profiler.summary['_define_dist_graph_comm'] + \\\n" + \
    "            sim.profiler.summary['_allocate_memory'] + \\\n" + \
    "            sim.profiler.summary['_load_initial_conditions'])\n" + \
    "        interior_times[jj,kk] = \\\n" + \
    "            sim.profiler.summary['run_interior_step'] / sim.settings.time_steps\n" + \
    "        boundary_times[jj,kk] = \\\n" + \
    "            (sim.profiler.summary['run_general_junction'] + \\\n" + \
    "            sim.profiler.summary['run_valve_step'] + \\\n" + \
    "            sim.profiler.summary['run_pump_step']) / sim.settings.time_steps\n" + \
    "        comm_times[jj,kk] = \\\n" + \
    "            (sim.profiler.summary['exchange_data'] + \\\n" + \
    "            sim.profiler.summary['barrier1'] + \\\n" + \
    "            sim.profiler.summary['barrier2']) / sim.settings.time_steps\n" + \
    "        totals = init_times + interior_times + boundary_times + comm_times\n" + \
    "    times = {\n" + \
    "        'init_times' : init_times,\n" + \
    "        'interior_times' : interior_times,\n" + \
    "        'boundary_times' : boundary_times,\n" + \
    "        'comm_times' : comm_times,\n" + \
    "        'totals' : totals,\n" + \
    f"        'processors' : processors,\n" + \
    f"        'time_steps' : time_steps\n" + \
    "    }\n" + \
    f"    with open('{export_path}', 'wb') as f:\n" + \
    "        pickle.dump(times, f)\n"
    if plot:
        fcontent += \
        "# -----------------------------------------------------\n" + \
        "# | Execute after running the bash generated by\n" + \
        "# | compute_simulation_times_per_step\n" + \
        "# v\n" + \
        "from ptsnet.graphics.static import plot_times_per_step\n" + \
        f"plot_times_per_step(duration={duration})\n"

    with open(compute_file_path, "w") as f:
        f.write(fcontent)

    print("\nExecute the following command on your terminal:")
    print(f"bash {bash_path}\n")
    exit()