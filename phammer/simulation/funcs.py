from numba import jit, vectorize
from phammer.simulation.constants import G, PARALLEL
import numpy as np

# ------------------ SIM STEPS ------------------

@jit(nopython = True, cache = True, parallel = PARALLEL)
def run_interior_step(Q0, H0, Q1, H1, B, R, Cp, Bp, Cm, Bm,
    is_pboundary, is_mboundary):
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
        Cm[i] = (H0[i+1] - B[i]*Q0[i+1]) * is_mboundary[i]
        Bm[i] = (B[i] + R[i]*abs(Q0[i+1])) * is_mboundary[i]
        Cp[i] = (H0[i-1] + B[i]*Q0[i-1]) * is_pboundary[i]
        Bp[i] = (B[i] + R[i]*abs(Q0[i-1])) * is_pboundary[i]
        H1[i] = (Cp[i]*Bm[i] + Cm[i]*Bp[i]) / (Bp[i] + Bm[i])
        Q1[i] = (Cp[i] - Cm[i]) / (Bp[i] + Bm[i])

# @jit(nopython = True, cache = True, nogil=True, parallel=True)
def run_boundary_step(
    H0, Q1, H1, E1, D1, Cp, Bp, Cm, Bm, Ke, Kd,
    mboundary_ids, pboundary_ids, reservoir_ids, jboundary_ids, jnode_ids, head_reps, bindices):
    """Solves flow and head for boundary points attached to nodes

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
        TODO: UPDATE ARGUMENTS
    """
    Cm[mboundary_ids] /= Bm[mboundary_ids]
    Cp[pboundary_ids] /= Bp[pboundary_ids]
    Bm[mboundary_ids] = 1 / Bm[mboundary_ids]
    Bp[pboundary_ids] = 1 / Bp[pboundary_ids]

    sc = np.add.reduceat(Cm[jboundary_ids], bindices) + np.add.reduceat(Cp[jboundary_ids], bindices)
    sb = np.add.reduceat(Bm[jboundary_ids], bindices) + np.add.reduceat(Bp[jboundary_ids], bindices)

    Z =  sc / sb
    K = ((Ke[jnode_ids] + Kd[jnode_ids])/sb)**2
    HH = ((2*Z + K) - np.sqrt(K**2 + 4*Z*K)) / 2
    H1[jboundary_ids] = HH[head_reps]
    H1[reservoir_ids] = H0[reservoir_ids]
    E1[jnode_ids] = Ke[jnode_ids] * np.sqrt(2*G*HH)
    D1[jnode_ids] = Kd[jnode_ids] * np.sqrt(2*G*HH)
    Q1[mboundary_ids] = H1[mboundary_ids]*Bm[mboundary_ids] - Cm[mboundary_ids]
    Q1[pboundary_ids] = Cp[pboundary_ids] - H1[pboundary_ids]*Bp[pboundary_ids]

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