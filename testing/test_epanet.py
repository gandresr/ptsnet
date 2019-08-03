import os

import phammer.epanet.toolkit as EPANET
from phammer.epanet.util import EN, FlowUnits
import matplotlib.pyplot as plt

inpfile = '/home/watsup/Documents/Github/phammer/example_files/PHFC_SIM_17_4_13.inp'
file_prefix, file_ext = os.path.splitext(inpfile)
rptfile = file_prefix + '.rpt'
outfile = file_prefix + '.bin'



EPANET_ = EPANET.ENepanet()
EPANET_.ENopen(inpfile, rptfile, outfile)

flow_units = FlowUnits(EPANET_.ENgetflowunits())


EPANET_.ENopenH()
EPANET_.ENinitH(0)

num_nodes = EPANET_.ENgetcount(EN.NODECOUNT)
num_links = EPANET_.ENgetcount(EN.LINKCOUNT)

link_name_list = []
node_name_list = []

for i in range(1, num_nodes+1):
    node_name_list.append(EPANET_.ENgetnodetype(i))
for i in range(1, num_links+1):
    link_name_list.append(EPANET_.ENgetlinktype(i))

flowrate = []
head = []

t = 0; period = 100
while EPANET_.ENnextH() > 0 and t <= period:
    tx = EPANET_.ENrunH()
    if t == period:
        for i in range(1, num_links+1):
            flowrate.append(FLOW_TO_SI * EPANET_.ENgetlinkvalue(i, EN.FLOW))
        for i in range(1, num_nodes+1):
            head.append(HEAD_TO_SI * EPANET_.ENgetnodevalue(i, EN.HEAD))
    t += 1

EPANET_.ENcloseH()
EPANET_.ENclose()

plt.plot(flowrate)
plt.show()