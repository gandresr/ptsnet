from ptsnet.simulation.constants import G
import numpy as np
from time import time

# ------------------ SIM STEPS ------------------

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

    # The first and last nodes are skipped in the  loop considering
    # that they are boundary nodes (every interior node requires an
    # upstream and a downstream neighbor)
    Cm[1:len(Q0)-1] = (H0[1+1:len(Q0)] - B[1:len(Q0)-1]*Q0[1+1:len(Q0)]) * has_minus[1:len(Q0)-1]
    Bm[1:len(Q0)-1] = (B[1:len(Q0)-1] + R[1:len(Q0)-1]*abs(Q0[1+1:len(Q0)])) * has_minus[1:len(Q0)-1]
    Cp[1:len(Q0)-1] = (H0[0:len(Q0)-2] + B[1:len(Q0)-1]*Q0[0:len(Q0)-2]) * has_plus[1:len(Q0)-1]
    Bp[1:len(Q0)-1] = (B[1:len(Q0)-1] + R[1:len(Q0)-1]*abs(Q0[0:len(Q0)-2])) * has_plus[1:len(Q0)-1]
    H1[1:len(Q0)-1] = (Cp[1:len(Q0)-1]*Bm[1:len(Q0)-1] + Cm[1:len(Q0)-1]*Bp[1:len(Q0)-1]) / (Bp[1:len(Q0)-1] + Bm[1:len(Q0)-1])
    Q1[1:len(Q0)-1] = (Cp[1:len(Q0)-1] - Cm[1:len(Q0)-1]) / (Bp[1:len(Q0)-1] + Bm[1:len(Q0)-1])

def run_boundary_step(H0, Q1, H1, E1, D1, Cp, Bp, Cm, Bm, Ke, Kd, Z, where):
    """Solves flow and head for boundary points attached to nodes

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
        TODO: UPDATE ARGUMENTS
    """
    Cm[where.points['jip_dboundaries']] /= Bm[where.points['jip_dboundaries']]
    Cp[where.points['jip_uboundaries']] /= Bp[where.points['jip_uboundaries']]
    Bm[where.points['jip_dboundaries']] = 1 / Bm[where.points['jip_dboundaries']]
    Bp[where.points['jip_uboundaries']] = 1 / Bp[where.points['jip_uboundaries']]

    sc = np.add.reduceat(Cm[where.points['just_in_pipes']] + Cp[where.points['just_in_pipes']]\
        , where.nodes['just_in_pipes',])
    sb = np.add.reduceat(Bm[where.points['just_in_pipes']] + Bp[where.points['just_in_pipes']]\
        , where.nodes['just_in_pipes',])

    X =  sc / sb
    K = ((Ke[where.nodes['all_just_in_pipes']] + Kd[where.nodes['all_just_in_pipes']])/sb)**2
    HZ = X - Z[where.nodes['all_just_in_pipes']]
    K[HZ < 0] = 0

    HH = ((2*X + K) - np.sqrt(K**2 + 4*K*HZ)) / 2
    H1[where.points['just_in_pipes']] = HH[where.points['just_in_pipes',]]
    H1[where.points['are_reservoirs']] = H0[where.points['are_reservoirs']]
    H1[where.points['are_tanks']] = H0[where.points['are_tanks']]
    Q1[where.points['jip_dboundaries']] = H1[where.points['jip_dboundaries']] \
        * Bm[where.points['jip_dboundaries']] - Cm[where.points['jip_dboundaries']]
    Q1[where.points['jip_uboundaries']] = Cp[where.points['jip_uboundaries']] \
        - H1[where.points['jip_uboundaries']] * Bp[where.points['jip_uboundaries']]

    # Get demand and leak flows
    HH -= Z[where.nodes['all_just_in_pipes']]
    HH[HH < 0] = 0 # No demand/leak flow with negative pressure
    num_jip = len(where.nodes['all_just_in_pipes'])
    E1[:num_jip] = Ke[where.nodes['all_just_in_pipes']] * np.sqrt(HH)
    D1[:num_jip] = Kd[where.nodes['all_just_in_pipes']] * np.sqrt(HH)

def run_valve_step(Q1, H1, Cp, Bp, Cm, Bm, setting, coeff, area, where):
    # --- End valves
    if len(where.points['are_single_valve',]) > 0:
        K0 = setting[where.points['are_single_valve',]] \
            * coeff[where.points['are_single_valve',]] \
                * area[where.points['are_single_valve',]]
        K = 2*G*(Bp[where.points['are_single_valve']] * K0)**2
        Cp_end = Cp[where.points['are_single_valve']]

        H1[where.points['are_single_valve']] =  \
            ((2*Cp_end + K) - np.sqrt((2*Cp_end + K)**2 - 4*Cp_end**2)) / 2

        Q1[where.points['are_single_valve']] = K0 * np.sqrt(2 * G * H1[where.points['are_single_valve']])

    # --- Inline valves
    if len(where.points['start_inline_valve']) > 0:
        CM = Cm[where.points['end_inline_valve']]
        BM = Bm[where.points['end_inline_valve']]
        CP = Cp[where.points['start_inline_valve']]
        BP = Bp[where.points['start_inline_valve']]

        S = np.sign(CP - CM)
        CV = 2 * G * (setting[where.points['start_inline_valve',]] \
            * coeff[where.points['start_inline_valve',]] \
                * area[where.points['start_inline_valve',]]) ** 2
        X = CV * (BP + BM)
        Q1[where.points['start_inline_valve']] = (-S*X + S*np.sqrt(X**2 + S*4*CV*(CP - CM)))/2
        Q1[where.points['end_inline_valve']] = Q1[where.points['start_inline_valve']]
        H1[where.points['start_inline_valve']] = CP - BP*Q1[where.points['start_inline_valve']]
        H1[where.points['end_inline_valve']] = CM + BM*Q1[where.points['start_inline_valve']]

def run_pump_step(source_head, Q1, H1, Cp, Bp, Cm, Bm, a1, a2, Hs, setting, where):
    if len(where.points['are_single_pump']) > 0:
        CP = source_head[where.points['are_single_pump',]]
        BP = np.zeros(len(where.points['are_single_pump']))
        CM = Cm[where.points['are_single_pump']]
        BM = Bm[where.points['are_single_pump']]

        alpha = setting[where.points['are_single_pump',]]
        A = a2[where.points['are_single_pump',]]
        B = a1[where.points['are_single_pump',]]*alpha - BM - BP
        C = Hs[where.points['are_single_pump',]]*alpha**2 - CM + CP

        root = B ** 2 - 4 * A * C; root[root < 0] = 0
        Q = (-B - np.sqrt(root)) / (2*A); Q[Q < 0] = 0
        Q1[where.points['are_single_pump']] = Q

        hp = a2[where.points['are_single_pump',]] * Q**2 + a1[where.points['are_single_pump',]]*alpha * Q + \
            Hs[where.points['are_single_pump',]]*alpha**2

        H1[where.points['are_single_pump']] =  CP + hp

    if len(where.points['start_inline_pump']) > 0:
        CP = Cp[where.points['start_inline_pump']]
        BP = Bp[where.points['start_inline_pump']]
        CM = Cm[where.points['end_inline_pump']]
        BM = Bm[where.points['end_inline_pump']]

        alpha = setting[where.points['start_inline_pump',]]
        A = a2[where.points['start_inline_pump',]]
        B = a1[where.points['start_inline_pump',]]*alpha - BM - BP
        C = Hs[where.points['start_inline_pump',]]*alpha**2 - CM + CP

        root = B ** 2 - 4 * A * C; root[root < 0] = 0
        Q = (-B - np.sqrt(root)) / (2*A); Q[Q < 0] = 0
        Q1[where.points['start_inline_pump']] = Q
        Q1[where.points['end_inline_pump']] = Q

        hp = a2[where.points['start_inline_pump',]] * Q**2 + a1[where.points['start_inline_pump',]]*alpha * Q + \
            Hs[where.points['start_inline_pump',]]*alpha**2

        H1[where.points['start_inline_pump']] =  CP - BP * Q
        H1[where.points['end_inline_pump']] =  H1[where.points['start_inline_pump']] + hp