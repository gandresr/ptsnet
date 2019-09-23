import os
import numpy as np

from wntr.epanet.io import InpFile
from phammer.arrays.arrays import Table
from phammer.epanet.toolkit import ENepanet
from phammer.epanet.util import EN, FlowUnits, HydParam, to_si
from phammer.simulation.constants import G, TOL, FLOOR_FFACTOR, CEIL_FFACTOR, DEFAULT_FFACTOR
from phammer.simulation.constants import NODE_PROPERTIES, PIPE_PROPERTIES, PUMP_PROPERTIES, VALVE_PROPERTIES

def get_water_network(inpfile):
    ENFile = InpFile()
    return ENFile.read(inpfile)

def get_initial_conditions(inpfile, period = 0, wn = None):

    # EPANET initialization

    file_prefix, _ = os.path.splitext(inpfile)
    rptfile = file_prefix + '.rpt'
    outfile = file_prefix + '.bin'

    if wn is None:
        wn = get_water_network(inpfile)

    network_graph = wn.get_graph()
    EPANET = ENepanet()
    EPANET.ENopen(inpfile, rptfile, outfile)
    EPANET.ENopenH()
    EPANET.ENinitH(0)

    # Data structures for node and link initial conditions
    nodes = Table(NODE_PROPERTIES, wn.num_nodes)
    node_ids = []
    pipes = Table(PIPE_PROPERTIES, wn.num_pipes)
    pipe_ids = []
    valves = Table(VALVE_PROPERTIES, wn.num_valves)
    valve_ids = []
    pumps = Table(PUMP_PROPERTIES, wn.num_pumps)
    pump_ids = []

    ic = {
        'node' : nodes,
        'pipe' : pipes,
        'valve' : valves,
        'pump' : pumps,
    }

    # Run EPANET simulation
    t = 0
    while EPANET.ENnextH() > 0 and t <= period: # EPS
        EPANET.ENrunH()
        t += 1
    if t == 0: # Not EPS
        EPANET.ENrunH()

    flow_units = FlowUnits(EPANET.ENgetflowunits())

    # Retrieve node conditions
    for i in range(1, wn.num_nodes+1):
        node_id = EPANET.ENgetnodeid(i)
        node_ids.append(node_id)
        ic['node'].leak_coefficient[i-1] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        ic['node'].demand[i-1] = EPANET.ENgetnodevalue(i, EN.DEMAND)
        ic['node'].head[i-1] = EPANET.ENgetnodevalue(i, EN.HEAD)
        ic['node'].pressure[i-1] = EPANET.ENgetnodevalue(i, EN.PRESSURE)
        ic['node'].type[i-1] = EPANET.ENgetnodetype(i)
        z = EPANET.ENgetnodevalue(i, EN.ELEVATION)
        if ic['node'].type[i-1] == EN.RESERVOIR:
            z = 0
        elif ic['node'].type[i-1] == EN.TANK:
            z = ic['node'].head[i-1] - ic['node'].pressure[i-1]
        ic['node'].elevation[i-1] = z
        ic['node'].degree[i-1] = network_graph.degree(node_id)

    # Unit conversion
    to_si(flow_units, ic['node'].leak_coefficient, HydParam.EmitterCoeff)
    to_si(flow_units, ic['node'].demand, HydParam.Flow)
    to_si(flow_units, ic['node'].head, HydParam.HydraulicHead)
    to_si(flow_units, ic['node'].pressure, HydParam.Pressure)
    to_si(flow_units, ic['node'].elevation, HydParam.Elevation)

    p, pp, v = 0, 0, 0 # pipes, pumps, valves
    for i in range(1, wn.num_links+1):

        link = wn.get_link(EPANET.ENgetlinkid(i))
        ltype = link.link_type.lower()

        if link.link_type == 'Pipe':
            k = p; p += 1
        elif link.link_type == 'Pump':
            k = pp; pp += 1
        elif link.link_type == 'Valve':
            k = v; v += 1

        ic[ltype].start_node[k], ic[ltype].end_node[k] = EPANET.ENgetlinknodes(i)
        ic[ltype].flowrate[k] = EPANET.ENgetlinkvalue(i, EN.FLOW)
        ic[ltype].velocity[k] = EPANET.ENgetlinkvalue(i, EN.VELOCITY)

        # Indexes are adjusted to fit the new Table / Indexing in EPANET's C code starts in 1
        ic[ltype].start_node[k] -= 1
        ic[ltype].end_node[k] -= 1

        if -TOL < ic[ltype].flowrate[k] < TOL:
            ic[ltype].direction[k] = 0
            ic[ltype].flowrate[k] = 0
            if link.link_type == 'Pipe':
                ic[ltype].ffactor[k] = DEFAULT_FFACTOR
        elif ic[ltype].flowrate[k] > TOL:
            ic[ltype].direction[k] = 1
        else:
            ic[ltype].direction[k] = -1
            ic[ltype].flowrate[k] *= -1
            ic[ltype].start_node[k], ic[ltype].end_node[k] = ic[ltype].end_node[k], ic[ltype].start_node[k]

        if ic['node'].degree[ic[ltype].start_node[k]] >= 2 and \
            ic['node'].degree[ic[ltype].end_node[k]] >= 2:
            ic[ltype].is_inline[k] = True

        if link.link_type in ('Pipe', 'Valve'):
            ic[ltype].diameter[k] = link.diameter
            ic[ltype].area[k] = np.pi * link.diameter ** 2 / 4
            ic[ltype].type[k] = EPANET.ENgetlinktype(i)

        if link.link_type == 'Pipe':
            pipe_ids.append(link.name)
            ic[ltype].length[k] = link.length
            ic[ltype].head_loss[k] = EPANET.ENgetlinkvalue(i, EN.HEADLOSS)
        elif link.link_type == 'Pump':
            pump_ids.append(link.name)
            ic[ltype].setting[k] = EPANET.ENgetlinkvalue(i, EN.SETTING)
            ic[ltype].initial_status[k] = link.initial_status
            # Pump curve parameters
            qp, hp = list(zip(*link.get_pump_curve().points)); qp = list(qp); hp = list(hp)
            qpp = to_si(flow_units, float(ic[ltype].flowrate[k]), HydParam.Flow)
            hpp = to_si(flow_units, EPANET.ENgetlinkvalue(i, EN.HEADLOSS), HydParam.HydraulicHead)
            qp.append(qpp); hp.append(abs(hpp))
            ic[ltype].a2[k], ic[ltype].a1[k], ic[ltype].Hs[k] = np.polyfit(qp, hp, 2)
            # Source head
            ic[ltype].source_head[k] = ic['node'].head[ic[ltype].start_node[k]]
        elif link.link_type == 'Valve':
            valve_ids.append(link.name)
            ic[ltype].initial_status[k] = EPANET.ENgetlinkvalue(i, EN.INITSTATUS)
            ic[ltype].setting[k] = ic[ltype].initial_status[k]
            ic[ltype].flowrate[k] = to_si(flow_units, float(ic[ltype].flowrate[k]), HydParam.Flow)
            ha = ic['node'].head[ic[ltype].start_node[k]]
            hb = ic['node'].head[ic[ltype].end_node[k]] if ic['node'].degree[ic[ltype].end_node[k]] > 1 else 0
            hl = ha - hb
            if hl > 0:
                ic[ltype].K[k] = ic[ltype].flowrate[k]/(ic[ltype].area[k]*(2*G*hl)**0.5)

    EPANET.ENcloseH()
    EPANET.ENclose()

    # Unit conversion
    to_si(flow_units, ic['pipe'].head_loss, HydParam.HydraulicHead)
    to_si(flow_units, ic['pipe'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['pump'].flowrate, HydParam.Flow)
    to_si(flow_units, ic['pipe'].velocity, HydParam.Velocity)
    to_si(flow_units, ic['pump'].velocity, HydParam.Velocity)
    to_si(flow_units, ic['valve'].velocity, HydParam.Velocity)

    idx = ic['pipe'].ffactor == 0
    ic['pipe'].ffactor[idx] = \
        (2*G*ic['pipe'].diameter[idx] * ic['pipe'].head_loss[idx]) \
            / (ic['pipe'].length[idx] * ic['pipe'].velocity[idx]**2)

    ic['pipe'].ffactor[ic['pipe'].ffactor >= CEIL_FFACTOR] = DEFAULT_FFACTOR
    ic['pipe'].ffactor[ic['pipe'].ffactor <= FLOOR_FFACTOR] = DEFAULT_FFACTOR

    ic['valve'].curve_index.fill(-1)
    ic['pump'].curve_index.fill(-1)

    demanded = np.logical_and(ic['node'].type != EN.RESERVOIR, ic['node'].type != EN.TANK)
    KeKd = ic['node'].demand[demanded] / np.sqrt(ic['node'].pressure[demanded])
    ic['node'].demand_coefficient[demanded] = KeKd - ic['node'].leak_coefficient[demanded]

    nodes.setindex(node_ids)
    pipes.setindex(pipe_ids)
    valves.setindex(valve_ids)
    pumps.setindex(pump_ids)

    return ic