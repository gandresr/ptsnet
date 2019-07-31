import os
import numpy as np
import phammer.epanet.toolkit.ENepanet as ENepanet

from phammer.arrays.table import Table
from phammer.epanet.util import EN, FlowUnits

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/PHFC_SIM_17_4_13.inp'

def get_initial_conditions(inpfile, period = 0):

    # EPANET initialization

    file_prefix, file_ext = os.path.splitext(inpfile)
    rptfile = file_prefix + '.rpt'
    outfile = file_prefix + '.bin'
    flow_units = FlowUnits(EPANET_.ENgetflowunits())

    EPANET = ENepanet()
    EPANET.ENopen(inpfile, rptfile, outfile)
    EPANET_.ENopenH()
    EPANET_.ENinitH(0)

    num_nodes = EPANET_.ENgetcount(EN.NODECOUNT)
    num_links = EPANET_.ENgetcount(EN.LINKCOUNT)

    # Data structures for node and link initial conditions

    nodes = Table({
        'ID' : '<U3',
        'emitter_coefficient' : np.float,
        'demand' : np.bool,
        'head' : np.float,
    }, num_nodes)

    links = Table({
        'ID' : '<U3'
        'flowrate' : np.float,
        'length' : np.float,
        'diameter' : np.float,
        'area' : np.float,
        'wave_speed' : np.float,
        'direction' : np.bool,
        'ffactor' : np.float,
        'start_node' : np.int,
        'end_node' : np.int
    }, num_links)

    conditions = {'links' : links, 'nodes' : nodes}

    # Run EPANET simulation
    t = 0
    while EPANET.ENnextH() > 0 and t <= period:
        tx = EPANET.ENrunH()
        t += 1

    # Retrieve node conditions
    for i in range(1, num_nodes+1):
        nodes.ID[i] = EPANET.ENgetnodeid(i)
        nodes.emitter_coefficient[i] = EPANET.ENgetnodevalue(i, EN.EMITTER)
        nodes.demand[i] = EPANET.ENgetnodevalue(i, EN.DEMAND)
        nodes.head[i] = EPANET.ENgetnodevalue(i, EN.HEAD)
    # Retrieve link conditions
    for i in range(1, num_links+1):
        links.ID[i] = EPANET.ENgetlinkid(i)
        links.flowrate[i] = EPANET.ENgetlinkvalue(i, EN.FLOW)
        links.length[i] = EPANET.ENgetlinkvalue(i, EN.LENGTH)
        links.diameter[i] = EPANET.ENgetlinkvalue(i, EN.DIAMETER)
        links.area[i] = EPANET.ENgetlinkvalue(i, np.pi * links.diameter[i] ** 2 / 4)
        links.direction[i] = EPANET.ENgetlinkvalue(i, EN.
        links.ffactor[i] = EPANET.ENgetlinkvalue(i, EN.
        links.start_node[i] = EPANET.ENgetlinkvalue(i, EN.
        links.end_node[i] = EPANET.ENgetlinkvalue(i, EN.

    EPANET_.ENcloseH()
    EPANET_.ENclose()

    return conditions

def get_network_graph():
    pass

def export_network_graph():
    pass