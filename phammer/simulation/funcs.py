from numba import jit
from phammer.simulation.constants import G, PARALLEL
import numpy as np

# ------------------ SIM STEPS ------------------

@jit(nopython = True, cache = True, parallel = PARALLEL)
def run_interior_step(Q0, H0, Q1, H1, B, R, Cp, Bp, Cm, Bm):
    """Solves flow and head for interior points

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    TODO: UPDATE ARGUMENTS
    """
    # Extreme points
    Cm[0] = H0[1] - B[0]*Q0[1]
    Cp[-1] = H0[-2] + B[-1]*Q0[-2]
    Bm[0] = B[0] + R[0]*abs(Q0[1])
    Bp[-1] = B[-1] + R[-1]*abs(Q0[-2])

    for i in range(1, len(Q0)-1):
        # The first and last nodes are skipped in the  loop considering
        # that they are boundary nodes (every interior node requires an
        # upstream and a downstream neighbor)
        Cm[i] = H0[i+1] - B[i]*Q0[i+1]
        Cp[i] = H0[i-1] + B[i]*Q0[i-1]
        Bm[i] = B[i] + R[i]*abs(Q0[i+1])
        Bp[i] = B[i] + R[i]*abs(Q0[i-1])
        H1[i] = (Cp[i]*Bm[i] + Cm[i]*Bp[i]) / (Bp[i] + Bm[i])
        Q1[i] = (H1[i] - Cm[i]) / Bm[i]

def run_junction_step(
    Q0, H0, Q1, H1, E1, D1, B, R, Cp, Bp, Cm, Bm,
    num_nodes, nodes_type, nodes_float, nodes_obj, RESERVOIR, JUNCTION):
    """Solves flow and head for boundary points attached to nodes

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
        TODO: UPDATE ARGUMENTS
        junctions_int {2D array} -- table with junction properties (integers)
        Q1 {array} -- initial flow
        Q2 {array} -- flow solution for the next time step
        H1 {array} -- initial head
        H2 {array} -- head solution for the next time step
        B {array} -- coefficients B[i] = a/(g*A)
        R {array} -- coefficients R[i] = f*dx/(2*g*D*A**2)
        junction_type {array} -- array with constants associated to junction type
        demand {array} -- demands at junctions
        emitter_coeff {array} -- K coefficients of emitters, Q = K(2gH)**0.5
        emitter_setting {array} -- setting value emitter in [0, 1]
        RESERVOIR {int} -- constant for junctions of type 'reservoir'
        PIPE {int} -- constant for junctions of type 'pipe'
        EMITTER {int} -- constant for junctions of type 'emitter'
        UP {int} -- row index in junctions_int to extract upstream_neighbors_num
        DOWN {int} -- row index in junctions_int to extract downstream_neighbors_num
        N {int} -- row index to extract the first downstream node in table junctions_int
    """
    for node_id in range(num_nodes):

        dpoints = nodes_obj.downstream_points[node_id]
        upoints = nodes_obj.upstream_points[node_id]

        # Junction is a reservoir
        # TODO : INCLUDE EMITTER
        if nodes_type[node_id] == RESERVOIR:
            Ke = nodes_float.emitter_setting[node_id]*nodes_float.emitter_coeff[node_id]
            Kd = nodes_float.demand_coeff[node_id]
            for k in dpoints:
                H1[k] = H0[k]
                Q1[k] = (H0[k] - Cm[k]) / Bm[k]
            for k in upoints:
                H1[k] = H0[k]
                Q1[k] = (Cp[k] - H0[k]) / Bp[k]
        if nodes_type[node_id] == JUNCTION:
            sc = 0
            sb = 0

            for k in dpoints: # C-
                sc += Cm[k] / Bm[k]
                sb += 1 / Bm[k]

            for k in upoints: # C+
                sc += Cp[k] / Bp[k]
                sb += 1 / Bp[k]

            Z = sc/sb
            Ke = nodes_float.emitter_setting[node_id]*nodes_float.emitter_coeff[node_id]
            Kd = nodes_float.demand_coeff[node_id]
            K = ((Ke+Kd)/sb)**2
            HH = ((2*Z + K) - (K**2 + 4*Z*K)**0.5) / 2

            E1[node_id] = Ke*(2*G*HH)**0.5
            D1[node_id] = Kd*(2*G*HH)**0.5

            for k in dpoints: # C-
                H1[k] = HH
                Q1[k] = (HH - Cm[k]) / Bm[k]

            for k in upoints: # C+
                H1[k] = HH
                Q1[k] = (Cp[k] - HH) / Bp[k]

@jit(nopython = True, cache = True, parallel = PARALLEL)
def run_valve_step(Q0, H0, Q1, H1, B, R, valves_int, valves_float, nodes_obj):
    for v in range(valves_int.shape[0]):
        start_id = valves_int.upstream_node[v]
        end_id = valves_int.downstream_node[v]
        unode = nodes_obj.upstream_points[start_id][0]
        dnode = nodes_obj.downstream_points[end_id]
        setting = valves_float.setting[v]
        valve_coeff = valves_float.valve_coeff[v]
        area = valves_float.area[v]
        Cp = H0[unode-1] + B[unode-1]*Q0[unode-1]
        Bp = B[unode-1] + R[unode-1]*abs(Q0[unode-1])

        if len(dnode) == 0:
            # End-valve
            K = 2*G*(Bp * setting * valve_coeff * area)**2
            H1[unode] = ((2*Cp + K) - ((2*Cp + K)**2 - 4*Cp**2) ** 0.5) / 2
            Q1[unode] = setting * valve_coeff * area * (2*G*H1[unode])
        else:
            dnode = dnode[0]
            # Inline-valve
            Cm = H0[dnode+1] - B[dnode+1]*Q0[dnode+1]
            Bm = B[dnode+1] + R[dnode+1]*abs(Q0[dnode+1])
            S = -1 if (Cp - Cm) < 0 else 1
            Cv = 2*G*(valve_coeff*setting*area)**2
            X = Cv*(Bp + Bm)
            Q1[unode] = (-S*X + S*(X**2 + S*4*Cv*(Cp - Cm))**0.5)/2
            Q1[dnode] = Q1[unode]
            H1[unode] = Cp - Bp*Q1[unode]
            H1[dnode] = Cm + Bm*Q1[dnode]