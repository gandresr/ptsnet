import wntr
from wntr.epanet.toolkit import ENepanet
from wntr.epanet.util import EN, FlowUnits, HydParam, to_si

from phammer.simulation.init import get_initial_conditions

inpfile = '/home/griano/Documents/Github/phammer/example_files/LoopedNet_leak_us.inp'
rptfile = '/home/griano/Documents/Github/phammer/example_files/LoopedNet_leak_us.rpt'
outfile = '/home/griano/Documents/Github/phammer/example_files/LoopedNet_leak_us.out'

wn = wntr.network.WaterNetworkModel(inpfile)

# Getting results from EPANET using WNTR

sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()
epa_result = float(results.node['demand']['5'])

# Extracting results with WNTR

EPANET = ENepanet()
EPANET.ENopen(inpfile, rptfile, outfile)
EPANET.ENopenH()
EPANET.ENinitH(0)
EPANET.ENrunH()

flow_units = FlowUnits(EPANET.ENgetflowunits())
emitter_coeff = to_si(flow_units, EPANET.ENgetnodevalue(5, EN.EMITTER), HydParam.EmitterCoeff)
pressure = to_si(flow_units, EPANET.ENgetnodevalue(5, EN.PRESSURE), HydParam.Pressure)
wntr_result = emitter_coeff * pressure ** 0.5

EPANET.ENcloseH()
EPANET.ENclose()

# Extracting results with phammer

ic = get_initial_conditions(inpfile)
K = ic['node'].leak_coefficient['5']
P = ic['node'].pressure['5']
H = ic['node'].head['5']
F = ic['node'].demand['5']
phammer_result = K*P**0.5

# Results Comparison
print("EPA", epa_result)
print("PHAMMER", phammer_result)
print("WNTR", wntr_result)