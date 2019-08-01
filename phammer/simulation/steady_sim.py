import os
import numpy as np

from wntr.epanet.io import InpFile
from phammer.arrays.table import Table
from phammer.epanet.toolkit import ENepanet
from phammer.epanet.util import EN, FlowUnits, HydParam, to_si
from phammer.simulation.constants import G, TOL, FLOOR_FFACTOR, CEIL_FFACTOR, DEFAULT_FFACTOR
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

    ic = {
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
        ic['nodes'].ID[i-1] = EPANET.ENgetnodeid(i)
        ic['nodes'].emitter_coefficient[i-1] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        ic['nodes'].demand[i-1] = EPANET.ENgetnodevalue(i, EN.DEMAND)
        ic['nodes'].head[i-1] = EPANET.ENgetnodevalue(i, EN.HEAD)

    # Retrieve link conditions
    head_loss = np.zeros(wn.num_links, dtype = np.float)

    p, pp, v = 0, 0, 0 # pipes, pumps, valves
    for i in range(1, wn.num_links+1):

        link = wn.get_link(EPANET.ENgetlinkid(i))
        ltype = link.link_type.lower() + 's'
        if link.link_type == 'Pipe':
            k = p
            p += 1
            ic[ltype].length[k] = link.length
            ic[ltype].head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)
        elif link.link_type == 'Pump':
            k = pp
            pp += 1
        elif link.link_type == 'Valve':
            k = v
            v += 1

        if link.link_type in ('Pipe', 'Valve'):
            ic[ltype].diameter[k] = link.diameter
            ic[ltype].area[k] = np.pi * link.diameter ** 2 / 4

        ic[ltype].ID[k] = link.name
        ic[ltype].start_node[k], ic[ltype].end_node[k] = EPANET.ENgetlinknodes(i)
        ic[ltype].flowrate[k] = EPANET.ENgetlinkvalue(i, EN.FLOW)
        ic[ltype].velocity[k] = EPANET.ENgetlinkvalue(i, EN.VELOCITY)

        if -TOL < ic[ltype].flowrate[k] < TOL:
            ic[ltype].direction[k] = 0
            ic[ltype].flowrate[k] = 0
            if link.link_type == 'Pipe':
                ic[ltype].ffactor[k] = DEFAULT_FFACTOR
        elif ic[ltype].flowrate[k] > TOL:
            ic[ltype].direction[k] = 1
        else:
            ic[ltype].direction[k] = -1

    EPANET.ENcloseH()
    EPANET.ENclose()

    # Unit conversion
    to_si(flow_units, ic['nodes'].emitter_coefficient, HydParam.EmitterCoeff)
    to_si(flow_units, ic['nodes'].demand, HydParam.Flow)
    to_si(flow_units, ic['nodes'].head, HydParam.HydraulicHead)
    to_si(flow_units, ic['pipes'].head_loss, HydParam.HeadLoss)
    to_si(flow_units, ic['pipes'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['pumps'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['valves'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['pipes'].velocity, HydParam.Velocity)
    to_si(flow_units, ic['pumps'].velocity, HydParam.Velocity)
    to_si(flow_units, ic['valves'].velocity, HydParam.Velocity)

    idx = ic['pipes'].ffactor == 0
    ic['pipes'].ffactor[idx] = \
        (2*G*ic['pipes'].diameter[idx] * ic['pipes'].head_loss[idx]) \
            / (ic['pipes'].length[idx] * ic['pipes'].velocity[idx]**2)

    ic['pipes'].ffactor[ic['pipes'].ffactor >= CEIL_FFACTOR] = DEFAULT_FFACTOR
    ic['pipes'].ffactor[ic['pipes'].ffactor <= FLOOR_FFACTOR] = DEFAULT_FFACTOR

    return ic

def get_network_graph():
    pass

def export_network_graph():
    pass