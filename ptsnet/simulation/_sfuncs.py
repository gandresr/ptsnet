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
    for i in range(1, len(Q0)-1):
        Cm[i] = (H0[i+1] - B[i]*Q0[i+1]) * has_minus[i]
        Bm[i] = (B[i] + R[i]*abs(Q0[i+1])) * has_minus[i]
        Cp[i] = (H0[i-1] + B[i]*Q0[i-1]) * has_plus[i]
        Bp[i] = (B[i] + R[i]*abs(Q0[i-1])) * has_plus[i]
        H1[i] = (Cp[i]*Bm[i] + Cm[i]*Bp[i]) / (Bp[i] + Bm[i])
        Q1[i] = (Cp[i] - Cm[i]) / (Bp[i] + Bm[i])

def run_boundary_step(H0, Q1, H1, E1, D1, Cp, Bp, Cm, Bm, Ke, Kd, Z, where):
    """Solves flow and head for boundary points attached to nodes

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
        TODO: UPDATE ARGUMENTS
    """
    for dboundary in where.points['jip_dboundaries']:
        Cm[dboundary] /= Bm[dboundary]
        Bm[dboundary] = 1 / Bm[dboundary]
    for uboundary in where.points['jip_uboundaries']:
        Cp[uboundary] /= Bp[uboundary]
        Bp[uboundary] = 1 / Bp[uboundary]

    X = np.zeros(len(where.nodes['just_in_pipes',]))
    sb = np.zeros(len(where.nodes['just_in_pipes',]))
    for i in range(len(where.nodes['just_in_pipes',])):
        j1 = where.nodes['just_in_pipes',][i]
        if i+1 == len(where.nodes['just_in_pipes',]):
            j2 = None
        else:
            j2 = where.nodes['just_in_pipes',][i+1]
        sc = sum(Cm[where.points['just_in_pipes']][j1:j2]) + \
                sum(Cp[where.points['just_in_pipes']][j1:j2])
        sb[i] = sum(Bm[where.points['just_in_pipes']][j1:j2]) + \
                sum(Bp[where.points['just_in_pipes']][j1:j2])
        X[i] =  sc / sb[i]

    HH = np.zeros(len(where.nodes['all_just_in_pipes']))
    for j, i in enumerate(where.nodes['all_just_in_pipes']):
        K = ((Ke[i] + Kd[i])/sb[j])**2
        HZ = X[j] - Z[i]
        if HZ < 0:
            K = 0
        HH[j] = ((2*X[j] + K) - np.sqrt(K**2 + 4*K*HZ)) / 2

    for j, i in enumerate(where.points['just_in_pipes',]):
        H1[where.points['just_in_pipes'][j]] = HH[i]
    for i in where.points['are_reservoirs']:
        H1[i] = H0[i]
    for i in where.points['are_tanks']:
        H1[i] = H0[i]
    for i in where.points['jip_dboundaries']:
        Q1[i] = H1[i] * Bm[i] - Cm[i]
    for i in where.points['jip_uboundaries']:
        Q1[i] = Cp[i] - H1[i] * Bp[i]

    # Get demand and leak flows
    for j, i in enumerate(where.nodes['all_just_in_pipes']):
        HH[j] -= Z[i]
        if HH[j] < 0:
            HH[j] = 0 # No demand/leak flow with negative pressure
        E1[j] = Ke[i] * np.sqrt(HH[j])
        D1[j] = Kd[i] * np.sqrt(HH[j])

def run_valve_step(Q1, H1, Cp, Bp, Cm, Bm, setting, coeff, area, where):
    # --- End valves
    ll = len(where.points['are_single_valve',])
    if ll > 0:
        K0 = np.zeros(ll)
        for i in where.points['are_single_valve',]:
            K0[i] = setting[i] * coeff[i] * area[i]
        for i, j in enumerate(where.points['are_single_valve']):
            K = 2*G*(Bp[j] * K0[i])**2
            H1[j] = ((2*Cp[j] + K) - np.sqrt((2*Cp[j] + K)**2 - 4*Cp[j]**2)) / 2
            Q1[j] = K0[i] * np.sqrt(2 * G * H1[j])

    # --- Inline valves
    ll = len(where.points['start_inline_valve'])
    if ll > 0:
        for i, h in enumerate(where.points['start_inline_valve',]):
            j = where.points['start_inline_valve'][i]
            k = where.points['end_inline_valve'][i]

            S = np.sign(Cp[j] - Cm[k])
            CV = 2 * G * (setting[h] * coeff[h] * area[h]) ** 2
            X = CV * (Bp[j] + Bm[k])
            Q1[j] = (-S*X + S*np.sqrt(X**2 + S*4*CV*(Cp[j] - Cm[k])))/2
            Q1[k] = Q1[j]
            H1[j] = Cp[j] - Bp[j]*Q1[j]
            H1[k] = Cm[k] + Bm[k]*Q1[j]

def run_pump_step(source_head, Q1, H1, Cp, Bp, Cm, Bm, a1, a2, Hs, setting, where):
    ll = len(where.points['are_single_pump'])
    if ll > 0:
        for i, j in enumerate(where.points['are_single_pump',]):
            k = where.points['are_single_pump'][i]
            BP = 0
            A = a2[j]
            B = a1[j]*setting[j] - Bm[k] - BP
            C = Hs[j]*setting[j]**2 - Cm[k] + source_head[j]

            root = B ** 2 - 4 * A * C
            if root < 0: root = 0
            Q = (-B - np.sqrt(root)) / (2*A)
            if Q < 0: Q = 0
            Q1[k] = Q

            hp = a2[j] * Q**2 + a1[j]*setting[j] * Q + Hs[j]*setting[j]**2

            H1[k] =  source_head[j] + hp

    ll = len(where.points['start_inline_pump'])
    if ll > 0:
        for i, h in enumerate(where.points['start_inline_pump',]):
            j = where.points['start_inline_pump'][i]
            k = where.points['end_inline_pump'][i]

            alpha = setting[h]
            A = a2[h]
            B = a1[h]*alpha - Bm[k] - Bp[j]
            C = Hs[h]*alpha**2 - Cm[k] + Cp[j]

            root = B ** 2 - 4 * A * C
            if root < 0: root = 0
            Q = (-B - np.sqrt(root)) / (2*A)
            if Q < 0: Q = 0
            Q1[j] = Q; Q1[k] = Q

            hp = a2[h] * Q**2 + a1[h]*alpha * Q + Hs[h]*alpha**2

            H1[j] =  Cp[j] - Bp[j] * Q
            H1[k] =  H1[j] + hp