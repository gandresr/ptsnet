import numpy as np
import mesh.Mesh as Mesh

class Simulation:
    """
    Here all the tables and properties required to
    run a MOC simulation are defined. Tables for
    simulations in parallel are created

    In the meantime:
    * valves are not valid between two junctions
    * it is not possible to connect one valve to another
    """
    def __init__(self, inp_file, duration, time_step):
        self.time_steps = T
        self.t = 0
        self.mesh = mesh
        self.steady_state_sim = mesh.steady_state_sim
        self.flow_results = np.zeros((T, mesh.num_nodes), dtype='float64')
        self.head_results = np.zeros((T, mesh.num_nodes), dtype='float64')
        self.curves = []
        self.settings = []
        self.define_initial_conditions()