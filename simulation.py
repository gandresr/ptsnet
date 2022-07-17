from ptsnet.simulation.sim import PTSNETSimulation
from ptsnet.utils.io import get_example_path

sim = PTSNETSimulation(
  workspace_name = 'TNET3_VALVE',
  inpfile = get_example_path('TNET3'),
  settings = {
    'show_progress' : True,
    'warnings_on' : False})
sim.define_valve_operation('VALVE-179', initial_setting=1, final_setting=0, start_time=1, end_time=2)
sim.run()