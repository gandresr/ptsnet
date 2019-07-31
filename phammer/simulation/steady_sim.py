import os
import numpy as np

from phammer.arrays.table import Table
from phammer.epanet.toolkit import ENepanet
from phammer.epanet.util import EN, FlowUnits, HydParam, to_si
from phammer.simulation.constants import G, TOL, DEFAULT_FFACTOR
from phammer.simulation.constants import NODE_INITIAL_CONDITIONS, LINK_INITIAL_CONDITIONS

def get_initial_conditions(inpfile, period = 0):

    # EPANET initialization

    file_prefix, file_ext = os.path.splitext(inpfile)
    rptfile = file_prefix + '.rpt'
    outfile = file_prefix + '.bin'

    EPANET = ENepanet()
    EPANET.ENopen(inpfile, rptfile, outfile)
    EPANET.ENopenH()
    EPANET.ENinitH(0)

    num_nodes = EPANET.ENgetcount(EN.NODECOUNT)
    num_links = EPANET.ENgetcount(EN.LINKCOUNT)

    # Data structures for node and link initial conditions
    nodes = Table(NODE_INITIAL_CONDITIONS, num_nodes)
    links = Table(LINK_INITIAL_CONDITIONS, num_links)
    conditions = {'links' : links, 'nodes' : nodes}

    # Run EPANET simulation
    t = 0
    while EPANET.ENnextH() > 0 and t <= period:
        tx = EPANET.ENrunH()
        t += 1

    flow_units = FlowUnits(EPANET.ENgetflowunits())

    # Retrieve node conditions
    for i in range(1, num_nodes+1):
        conditions['nodes'].ID[i-1] = EPANET.ENgetnodeid(i)
        conditions['nodes'].emitter_coefficient[i-1] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        conditions['nodes'].demand[i-1] = to_si(flow_units, EPANET.ENgetnodevalue(i, EN.DEMAND), HydParam.Flow)
        conditions['nodes'].head[i-1] = to_si(flow_units, EPANET.ENgetnodevalue(i, EN.HEAD), HydParam.HydraulicHead)
    # Retrieve link conditions
    for i in range(1, num_links+1):
        conditions['links'].ID[i-1] = EPANET.ENgetlinkid(i)
        conditions['links'].start_node[i-1], conditions['links'].end_node[i-1] = EPANET.ENgetlinknodes(i)
        conditions['links'].length[i-1] = to_si(flow_units, EPANET.ENgetlinkvalue(i, EN.LENGTH), HydParam.Length)
        conditions['links'].diameter[i-1] = to_si(flow_units, EPANET.ENgetlinkvalue(i, EN.DIAMETER), HydParam.PipeDiameter)
        conditions['links'].area[i-1] = np.pi * conditions['links'].diameter[i-1] ** 2 / 4
        conditions['links'].flowrate[i-1] = to_si(flow_units, EPANET.ENgetlinkvalue(i, EN.FLOW), HydParam.Flow)

        if conditions['links'].flowrate[i-1] > TOL:
            conditions['links'].direction[i-1] = 1
        elif -TOL < conditions['links'].flowrate[i-1] < TOL:
            conditions['links'].direction[i-1] = 0
            conditions['links'].flowrate[i-1] = 0
        else:
            conditions['links'].direction[i-1] = -1

        hl = to_si(flow_units, EPANET.ENgetlinkvalue(i, EN.HEADLOSS), HydParam.HeadLoss) # Head loss

        den = (conditions['links'].length[i-1]*(conditions['links'].flowrate[i-1]/conditions['links'].area[i-1])**2)
        if not (-TOL < den < TOL):
            conditions['links'].ffactor[i-1] = min(DEFAULT_FFACTOR, 2*G*conditions['links'].diameter[i-1]*hl / den )
        else:
            conditions['links'].ffactor[i-1] = DEFAULT_FFACTOR

    EPANET.ENcloseH()
    EPANET.ENclose()

    # Unit conversion
    conditions['nodes'].emitter_coefficient[:] = to_si(flow_units, conditions['nodes'].emitter_coefficient, HydParam.EmitterCoeff)

    return conditions

def get_network_graph():
    pass

def export_network_graph():
    pass