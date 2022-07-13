from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path

# ------------------ (1) Model Setup ------------------

default_settings = {
    "time_step" : 0.01, # Simulation time step in [s]
    "duration" : 20, # Simulation duration in [s]
    "period" : 0, # Simulation period for EPS
    "default_wave_speed" : 1000, # Wave speed value for all pipes in [m/s]
    "wave_speed_file_path" : None, # Text file with wave speed values
    "delimiter" : ',', # Delimiter of text file with wave speed values
    "wave_speed_method" : 'optimal', # Wave speed adjustment method
    "save_results" : True, # Saves numerical results in HDF5 format
    "skip_compatibility_check" : False, # Dismisses compatibility check
    "show_progress" : False, # Shows progress (Warnings should be off)
    "profiler_on" : False, # Measures computational times of the simulation
    "warnings_on" : False, # Warnings are displayed if True
}

# Create a simulation
sim = PTSNETSimulation(workspace_name = "TNET3_SIM",
    inpfile = get_example_path('TNET3'),
    settings = default_settings
    # If settings are not defined, default settings are loaded automatically
)

# Define transient scenario
sim.define_valve_operation('VALVE-179',
    initial_setting=1, final_setting=0, start_time=0, end_time=1)
# sim.define_valve_settings('VALVE-179',
#   X=[start_time, end_time], Y=[initial_setting, final_setting])

# ------------------ (2) Execution ------------------

sim.run()
# while not sim.is_over():
#     sim.run_step()

# ------------------ (3) Extraction ------------------

import matplotlib.pyplot as plt
from ptsnet.simulation.sim import PTSNETSimulation

with PTSNETSimulation("TNET3_SIM") as sim:
    plt.plot(sim['time'], sim['node'].head['JUNCTION-23'], label='JUNCTION-23')
    plt.xlabel('Time [s]'); plt.ylabel('Head [m]'); plt.legend()
    plt.show()