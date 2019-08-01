import os
import numpy as np

from wntr.epanet.io import InpFile
from phammer.arrays.table import Table
from phammer.epanet.toolkit import ENepanet
from phammer.epanet.util import EN, FlowUnits, HydParam, to_si
from phammer.simulation.constants import G, TOL, DEFAULT_FFACTOR
from phammer.simulation.constants import NODE_INITIAL_CONDITIONS, PIPE_INITIAL_CONDITIONS, PUMP_INITIAL_CONDITIONS, VALVE_INITIAL_CONDITIONS

def get_initial_conditions(inpfile, period = 0):

    # EPANET initialization

    file_prefix, file_ext = os.path.splitext(inpfile)
    rptfile = file_prefix + '.rpt'
    outfile = file_prefix + '.bin'

    ENFile = InpFile()
    wn = ENFile.read(inpfile)
    EPANET = ENepanet()
    EPANET.ENopen(inpfile, rptfile, outfile)
    EPANET.ENopenH()
    EPANET.ENinitH(0)

    # Data structures for node and link initial conditions
    nodes = Table(NODE_INITIAL_CONDITIONS, wn.num_nodes)
    pipes = Table(PIPE_INITIAL_CONDITIONS, wn.num_pipes)
    valves = Table(VALVE_INITIAL_CONDITIONS, wn.num_valves)
    pumps = Table(PUMP_INITIAL_CONDITIONS, wn.num_pumps)

    initial_conditions = {
        'nodes' : nodes,
        'pipes' : pipes,
        'valves' : valves,
        'pumps' : pumps,
    }

    # Run EPANET simulation
    t = 0
    while EPANET.ENnextH() > 0 and t <= period: # EPS
        EPANET.ENrunH()
        t += 1
    if t == 0: # Not EPS
        EPANET.ENrunH()

    flow_units = FlowUnits(EPANET.ENgetflowunits())

    # Selectors

    # 'are_junction'
    # 'are_tank'
    # 'are_reservoir'

    # Retrieve node conditions
    for i in range(1, wn.num_nodes+1):
        initial_conditions['nodes'].ID[i-1] = EPANET.ENgetnodeid(i)
        initial_conditions['nodes'].emitter_coefficient[i-1] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        initial_conditions['nodes'].demand[i-1] = EPANET.ENgetnodevalue(i, EN.DEMAND)
        initial_conditions['nodes'].head[i-1] = EPANET.ENgetnodevalue(i, EN.HEAD)

    # Retrieve link conditions
    head_loss = np.zeros(wn.num_links, dtype = np.float)

    p, pp, v = 0, 0, 0 # pipes, pumps, valves
    for i in range(1, wn.num_links+1):

        link = wn.get_link(EPANET.ENgetlinkid(i))
        ltype = link.link_type.lower() + 's'
        if link.link_type == 'Pipe':
            k = p
            p += 1
            initial_conditions[ltype].length[k] = link.length
        elif link.link_type == 'Pump':
            k = pp
            pp += 1
        elif link.link_type == 'Valve':
            k = v
            v += 1

        if link.link_type in ('Pipe', 'Valve'):
            initial_conditions[ltype].diameter[k] = link.diameter
            initial_conditions[ltype].area[k] = np.pi * link.diameter ** 2 / 4

        initial_conditions[ltype].ID[k] = link.name
        initial_conditions[ltype].start_node[k], initial_conditions[ltype].end_node[k] = EPANET.ENgetlinknodes(i)
        initial_conditions[ltype].flowrate[k] = EPANET.ENgetlinkvalue(i, EN.FLOW)

        if initial_conditions[ltype].flowrate[k] > TOL:
            initial_conditions[ltype].direction[k] = 1
        elif -TOL < initial_conditions[ltype].flowrate[k] < TOL:
            initial_conditions[ltype].direction[k] = 0
            initial_conditions[ltype].flowrate[k] = 0
        else:
            initial_conditions[ltype].direction[k] = -1

        head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)

    EPANET.ENcloseH()
    EPANET.ENclose()

    # Unit conversion
    to_si(flow_units, initial_conditions['nodes'].emitter_coefficient, HydParam.EmitterCoeff)
    to_si(flow_units, initial_conditions['nodes'].demand, HydParam.Flow)
    to_si(flow_units, initial_conditions['nodes'].head, HydParam.HydraulicHead)
    to_si(flow_units, initial_conditions['pipes'].flowrate, HydParam.Flow)
    to_si(flow_units, initial_conditions['pumps'].flowrate, HydParam.Flow)
    to_si(flow_units, initial_conditions['valves'].flowrate, HydParam.Flow)
    to_si(flow_units, head_loss, HydParam.HeadLoss)

    # den = (conditions['links'].length*(conditions['links'].flowrate/conditions['links'].area)**2)
    # conditions['links'].ffactor = min(DEFAULT_FFACTOR, 2*G*conditions['links'].diameter[i-1]*hl / den )

    # conditions['links'].R[:] =
    return initial_conditions, wn

def get_network_graph():
    pass

def export_network_graph():
    pass