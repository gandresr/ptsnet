import numpy as np
import os, psutil

from ptsnet.utils.tacc import create_tacc_job, submit_tacc_jobs
from ptsnet.simulation.constants import FILE_TEMPLATE
from ptsnet.simulation.sim import PTSNETSimulation, PTSNETSettings
from ptsnet.results.workspaces import new_uuid_workspace_name, create_temp_folder

def compute_num_processors(
    sim,
    plot = False,
    count = 4,
    max_num_processors = None,
    steps = 2500,
    environment = 'pc',
    queue = 'normal',
    processors_per_node = 64,
    run_time = 30, # minutes
    allocation = None):

    if environment not in ('pc', 'tacc'): raise ValueError("Environment can only be ('pc', 'tacc')")
    if environment == 'tacc' and not allocation: raise ValueError("Specify your TACC allocation")

    if type(sim) is not PTSNETSimulation: raise ValueError("'sim' must be a PTSNETSimulation")
    nprocessors = psutil.cpu_count(logical=False) if max_num_processors is None else max_num_processors
    temp_folder_path = create_temp_folder(root=True)
    workspaces = new_uuid_workspace_name(count)
    temp_file_paths = [os.path.join(temp_folder_path, f"temp_sim_{i}.py") for i in range(count)]
    compute_file_path = os.path.join(temp_folder_path, "compute_processors_sim.py")
    export_path = os.path.join(temp_folder_path, "exported_processor_times.pkl")

    # Modify settings for performance simulation only
    sim.settings.profiler_on = True
    sim.settings.duration = steps*sim.time_step
    sim.settings.save_results = True

    # Write Python script to run simulations
    for ii in range(count):
        with open(temp_file_paths[ii], "w") as f:
            f.write(FILE_TEMPLATE.format(workspace_name="tmp"+os.path.sep+f"{workspaces[ii]}", inpfile=sim.inpfile, settings=sim.settings.to_dict(simplified=True)))
            f.write("sim.run()")

    processors = np.linspace(1, nprocessors, count, dtype=int)

    # Write bash executable
    if environment == 'pc':
        bash_path = os.path.join(temp_folder_path, "compute_num_processors.sh")
        with open(bash_path, "w") as f:
            f.write(f"echo 'Evaluating Performance (this might take a few minutes)'\n")
            for ii, p in enumerate(processors):
                f.write(f"mpiexec -n {p} python3 {temp_file_paths[ii]} &> log.txt\n")
                f.write(f"echo '({int(100*(ii+1)/len(processors))}%) Finished run {ii+1}/{len(processors)}'\n")
            f.write(f"rm log.txt\n")
            f.write(f"python3 {compute_file_path}\n")
    elif environment == 'tacc':
        for ii, p in enumerate(processors):
            create_tacc_job(
                fpath = temp_file_paths[ii],
                job_name = f"j_{ii}",
                num_processors = p,
                allocation = allocation,
                run_time = run_time,
                queue = queue,
                processors_per_node = processors_per_node)
        submit_tacc_jobs()
        print('\n-----------------------------------------------------\n')
        print(f'{len(processors)} jobs have been submitted to TACC\n')
        print("After running your jobs ('squeue -u <user>') execute the following command on the terminal:")
        print(f"python3 {compute_file_path}")
        print('\n-----------------------------------------------------\n')

    # Write Python script to compute number of processors
    fcontent = "import matplotlib.pyplot as plt\n" if plot else ""
    fcontent += \
        "import os\n" + \
        "import pickle\n" + \
        "import numpy as np\n" + \
        "from pprint import pprint\n" + \
        "from ptsnet.simulation.sim import PTSNETSimulation\n" + \
        "from ptsnet.results.workspaces import get_tmp_folder\n" + \
        "from kneed import KneeLocator\n" + \
        f"workspaces = {workspaces}\n" + \
        "times = {}\n" + \
        "for w in workspaces:\n" + \
        f"    with PTSNETSimulation('tmp'+os.path.sep+w) as sim:\n" + \
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
        "print(f'\\n--> Recommended number of processors: {optimal}\\n')\n" + \
        f"with open('{export_path}', 'wb') as f:\n" + \
        "    pickle.dump({'processor': x, 'time': y, 'optimal': optimal}, f)\n"

    if plot:
        fcontent += \
        "from ptsnet.graphics.static import plot_knee\n" + \
        f"plot_knee()\n"
    else:
        fcontent += \
        "print('-----------------------------------------------------\\n')\n" + \
        'print("Average times per step' + " {'processor': x, 'time': y, 'optimal': optimal}" + ' have been pickled to:")\n' + \
        f"print('export_path = {export_path}\\n')\n" + \
        "print('-----------------------------------------------------\\n')\n"

    with open(compute_file_path, "w") as f:
        f.write(fcontent)

    if environment == 'pc':
        print("\nExecute the following commands on your terminal:")
        print(f"cd {os.getcwd()}")
        print(f"bash {bash_path}\n")

    return

def compute_simulation_times(
    inpfile,
    time_steps,
    duration = 20,
    plot = False,
    count = 4,
    max_num_processors = None,
    steps = 2500,
    environment = 'pc',
    queue = 'normal',
    processors_per_node = 64,
    run_time = 30, # minutes
    allocation = None,
    clean_files = True):

    if environment not in ('pc', 'tacc'): raise ValueError("Environment can only be ('pc', 'tacc')")
    if environment == 'tacc' and not allocation: raise ValueError("Specify your TACC allocation")

    temp_folder_path = create_temp_folder(root=True)
    nprocessors = psutil.cpu_count(logical=False) if max_num_processors is None else max_num_processors
    processors = np.linspace(1, nprocessors, count, dtype=int)
    compute_file_path = os.path.join(temp_folder_path, "compute_times_per_step.py")
    workspace_name = new_uuid_workspace_name(1)[0]
    sims = {(processors[ii], time_steps[jj]) : (ii, jj)  for ii in range(len(processors)) for jj in range(len(time_steps))}
    bash_path = os.path.join(temp_folder_path, "compute_times_per_step.sh")

    kk = 0
    for ii, ts in enumerate(time_steps):
        # Write Python script to run simulations
        settings = PTSNETSettings(
            time_step = ts,
            duration = steps*ts,
            warnings_on = False,
            show_progress = True,
            skip_compatibility_check = True,
            save_results = True,
            profiler_on = True,
            wave_speed_method = 'user'
        )
        for jj in range(count):
            temp_file_path = os.path.join(temp_folder_path, f"temp_sim_{ii}_{jj}.py")
            with open(temp_file_path, "w") as f:
                f.write(FILE_TEMPLATE.format(workspace_name='tmp'+os.path.sep+workspace_name+f'_{ii}_{jj}', inpfile=inpfile, settings=settings.to_dict(simplified=True)))
                f.write("sim.run()")
            kk += 1

    # Write bash executable
    if environment == 'pc':
        with open(bash_path, "w") as f:
            f.write(f"echo 'Executing Simulations to Evaluate Performance'\n")
            f.write(f"echo '[This might take a few minutes]'\n")
            for ii, (p, ts) in enumerate(sims):
                jj = list(time_steps).index(ts)
                kk = list(processors).index(p)
                temp_file_path = os.path.join(temp_folder_path, f"temp_sim_{jj}_{kk}.py")
                f.write(f"mpiexec -n {p} python3 {temp_file_path} &> log.txt\n")
                f.write(f"echo '({int(100*(1+ii)/len(sims))}%) Finished Simulation {1+ii}/{len(sims)} -> {p} processor(s) | time_step = {ts} s'\n")
            f.write(f"rm log.txt\n")
            f.write(f"python3 {compute_file_path}\n")
    elif environment == 'tacc':
        for ii, (p, ts) in enumerate(sims):
            jj = list(time_steps).index(ts)
            kk = list(processors).index(p)
            temp_file_path = os.path.join(temp_folder_path, f"temp_sim_{jj}_{kk}.py")
            create_tacc_job(
                fpath = temp_file_path,
                job_name = f"j_{ii}",
                num_processors = p,
                allocation = allocation,
                run_time = 30,
                processors_per_node = processors_per_node,
                queue = queue)
        submit_tacc_jobs()
        print('\n-----------------------------------------------------\n')
        print(f'{len(sims)} jobs have been submitted to TACC\n')
        print("After running your jobs ('squeue -u <user>') execute the following command on the terminal:")
        print(f"python3 {compute_file_path}")
        print('\n-----------------------------------------------------\n')

    # Write Python script to compute simulation times
    export_path = os.path.join(temp_folder_path, "exported_sim_times.pkl")
    fcontent = \
    "import numpy as np\n" + \
    "import os\n" + \
    "import pickle\n" + \
    "from ptsnet.simulation.sim import PTSNETSimulation\n" + \
    "from ptsnet.results.workspaces import get_num_tmp_workspaces\n" + \
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
    f"    workspace_name = 'tmp'+os.path.sep+'{workspace_name}'+f'_"+"{jj}_{kk}'\n" + \
    "    with PTSNETSimulation(workspace_name) as sim:\n" + \
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
        "from ptsnet.graphics.static import plot_estimated_simulation_times\n" + \
        f"plot_estimated_simulation_times(duration={duration})\n"
    else:
        fcontent += \
        "print('-----------------------------------------------------\\n')\n" + \
        "print('Simulation times have been exported to:')\n" + \
        f"print('export_path = {export_path}\\n')\n" + \
        "print('Plot your results executing:\\n')\n" + \
        "print('>>> from ptsnet.graphics.static import plot_estimated_simulation_times')\n" + \
        f"print('>>> plot_estimated_simulation_times(duration={duration}, fpath=export_path)\\n')\n" + \
        "print('-----------------------------------------------------\\n')\n"

    with open(compute_file_path, "w") as f:
        f.write(fcontent)

    if environment == 'pc':
        print("\nExecute the following commands on your terminal:")
        print(f"cd {os.getcwd()}")
        print(f"bash {bash_path}\n")
    return