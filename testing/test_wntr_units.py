import wntr
from phammer.simulation.ic import get_initial_conditions
inpfile = '/home/watsup/Documents/Github/phammer/example_files/LoopedNet_leak_us.inp'

wn = wntr.network.WaterNetworkModel(inpfile)
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
wntr_result = float(results.node['demand']['5'])
node = wn.get_node('5')

ic = get_initial_conditions(inpfile)
K = ic['nodes'].emitter_coefficient['5']
P = ic['nodes'].pressure['5']
H = ic['nodes'].head['5']
F = ic['nodes'].demand['5']
epa_result = K*P**0.5
print(wntr_result, epa_result)