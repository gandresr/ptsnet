

FT_TO_M = 0.3048
IN_TO_MM = 25.4

EPANET_ = EPANET.ENepanet()
EPANET_.ENopen(inpfile, rptfile, outfile)

flow_units = FlowUnits(EPANET_.ENgetflowunits())
HEAD_TO_SI = FT_TO_M if not flow_units.is_metric else 1
FLOW_TO_SI = flow_units.factor

EPANET_.ENopenH()
EPANET_.ENinitH(0)

num_nodes = EPANET_.ENgetcount(EN.NODECOUNT)
num_links = EPANET_.ENgetcount(EN.LINKCOUNT)

link_name_list = []
node_name_list = []

for i in range(1, num_nodes+1):
    node_name_list.append(EPANET_.ENgetnodeid(i))
for i in range(1, num_links+1):
    link_name_list.append(EPANET_.ENgetlinkid(i))

flowrate = []
head = []


    if t == period:
        for i in range(1, num_links+1):
            flowrate.append(FLOW_TO_SI * EPANET_.ENgetlinkvalue(i, EN.FLOW))
        for i in range(1, num_nodes+1):
            head.append(HEAD_TO_SI * EPANET_.ENgetnodevalue(i, EN.HEAD))

EPANET_.ENcloseH()
EPANET_.ENclose()

plt.plot(flowrate)
plt.show()