import wntr

input_file = 'example_files/LoopedNet_leak.inp'
wn = wntr.network.WaterNetworkModel(input_file)
wn.get_node('5').add_leak(wn, area=1, discharge_coeff=1/1000, start_time=0)
wn.get_node('5').remove_leak(wn)
steady_state_sim = wntr.sim.WNTRSimulator(wn, mode='DD').run_sim()
steady_state_sim = wntr.sim.EpanetSimulator(wn).run_sim()
print(steady_state_sim.link['flowrate']['5'])
print(steady_state_sim.node['demand']['5'])
print(steady_state_sim.node['leak_demand']['5'])