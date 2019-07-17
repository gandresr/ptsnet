import wntr

input_file = 'example_files/LoopedNet_leak.inp'
wn = wntr.network.WaterNetworkModel(input_file)
steady_state_sim = wntr.sim.EpanetSimulator(wn).run_sim()
print(steady_state_sim.node['demand']['5'])