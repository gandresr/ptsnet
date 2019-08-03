from numba import jit, vectorize
from phammer.simulation.constants import G, PARALLEL
import numpy as np

# ------------------ SIM STEPS ------------------

@jit(nopython = True, cache = True, parallel = PARALLEL)
def run_interior_step(Q0, H0, Q1, H1, B, R, Cp, Bp, Cm, Bm,
    has_plus, has_minus):
    """Solves flow and head for interior points

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    TODO: UPDATE ARGUMENTS
    """
    # Extreme points
    Cm[0] = H0[1] - B[0]*Q0[1]
    Cp[-1] = H0[-2] + B[-2]*Q0[-2]
    Bm[0] = B[0] + R[0]*abs(Q0[1])
    Bp[-1] = B[-2] + R[-2]*abs(Q0[-2])

    for i in range(1, len(Q0)-1):
        # The first and last nodes are skipped in the  loop considering
        # that they are boundary nodes (every interior node requires an
        # upstream and a downstream neighbor)
        Cm[i] = (H0[i+1] - B[i]*Q0[i+1]) * has_minus[i]
        Bm[i] = (B[i] + R[i]*abs(Q0[i+1])) * has_minus[i]
        Cp[i] = (H0[i-1] + B[i]*Q0[i-1]) * has_plus[i]
        Bp[i] = (B[i] + R[i]*abs(Q0[i-1])) * has_plus[i]
        H1[i] = (Cp[i]*Bm[i] + Cm[i]*Bp[i]) / (Bp[i] + Bm[i])
        Q1[i] = (Cp[i] - Cm[i]) / (Bp[i] + Bm[i])

# @jit(nopython = True, cache = True, nogil=True, parallel=True)
def run_boundary_step(H0, Q1, H1, E1, D1, Cp, Bp, Cm, Bm, Ke, Kd, where):
    """Solves flow and head for boundary points attached to nodes

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
        TODO: UPDATE ARGUMENTS
    """
    Cm[where.points['are_dboundaries']] /= Bm[where.points['are_dboundaries']]
    Cp[where.points['are_uboundaries']] /= Bp[where.points['are_uboundaries']]
    Bm[where.points['are_dboundaries']] = 1 / Bm[where.points['are_dboundaries']]
    Bp[where.points['are_uboundaries']] = 1 / Bp[where.points['are_uboundaries']]

    sc = np.add.reduceat(Cm[where.points['are_junctions']], where.points['are_junctions',]) + \
        np.add.reduceat(Cp[where.points['are_junctions']], where.points['are_junctions',])
    sb = np.add.reduceat(Bm[where.points['are_junctions']], where.points['are_junctions',]) + \
        np.add.reduceat(Bp[where.points['are_junctions']], where.points['are_junctions',])

    Z =  sc / sb
    K = ((Ke[jnode_ids] + Kd[jnode_ids])/sb)**2
    HH = ((2*Z + K) - np.sqrt(K**2 + 4*Z*K)) / 2
    H1[where.points['are_junctions']] = HH[where.points['to_junctions']]
    H1[where.points['are_reservoirs']] = H0[where.points['are_reservoirs']]
    E1[jnode_ids] = Ke[jnode_ids] * np.sqrt(2*G*HH)
    D1[jnode_ids] = Kd[jnode_ids] * np.sqrt(2*G*HH)
    Q1[where.points['are_dboundaries']] = H1[where.points['are_dboundaries']] \
        * Bm[where.points['are_dboundaries']] - Cm[where.points['are_dboundaries']]
    Q1[where.points['are_pboundaries']] = Cp[where.points['are_pboundaries']] \
        - H1[where.points['are_pboundaries']] * Bp[where.points['are_pboundaries']]

@jit(nopython = True, cache = True, parallel = PARALLEL)
def run_valve_step(Q0, H0, Q1, H1, B, R, valves_int, valves_float, nodes_obj):
    for v in range(valves_int.shape[0]):
        start_ids = valves_int.upstream_node
        end_ids = valves_int.downstream_node
        unodes = node_points[start_ids, 0]
        dnodes = node_points[end_ids, 0]
        setting = valves_float.setting
        valve_coeff = valves_float.valve_coeff
        area = valves_float.area
        Cp = H0[unode-1] + B[unode-1]*Q0[unode-1]
        Bp = B[unode-1] + R[unode-1]*abs(Q0[unode-1])

        # End-valve
        K = 2*G*(Bp[unodes] * setting * valve_coeff * area)**2
        H1[unodes] = ((2*Cp[unodes] + K) - ((2*Cp[unodes] + K)**2 - 4*Cp[unodes]**2) ** 0.5) / 2
        Q1[unodes] = setting * valve_coeff * area * (2*G*H1[unodes])

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