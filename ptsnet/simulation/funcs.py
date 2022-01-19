from ptsnet.simulation.constants import G
import numpy as np
from time import time
from scipy import optimize
from numba import jit

# ------------------ SIM STEPS ------------------
@jit(nopython = True)
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

def run_general_junction(H0, Q1, H1, E1, D1, Cp, Bp, Cm, Bm, Ke, Kd, Z, where):
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
    if len(where.points['single_valve',]) > 0:
        K0 = setting[where.points['single_valve',]] \
            * coeff[where.points['single_valve',]] \
                * area[where.points['single_valve',]] # C = K*sqrt(2g \Delta H)
        K = 2*G*(Bp[where.points['single_valve']] * K0)**2
        Cp_end = Cp[where.points['single_valve']]

        H1[where.points['single_valve']] =  \
            ((2*Cp_end + K) - np.sqrt((2*Cp_end + K)**2 - 4*Cp_end**2)) / 2

        Q1[where.points['single_valve']] = K0 * np.sqrt(2 * G * H1[where.points['single_valve']])

    # --- Inline valves
    if len(where.points['start_valve']) > 0:
        CM = Cm[where.points['end_valve']]
        BM = Bm[where.points['end_valve']]
        CP = Cp[where.points['start_valve']]
        BP = Bp[where.points['start_valve']]

        S = np.sign(CP - CM)
        CV = 2 * G * (setting[where.points['start_valve',]] \
            * coeff[where.points['start_valve',]] \
                * area[where.points['start_valve',]]) ** 2
        X = CV * (BP + BM)
        Q1[where.points['start_valve']] = (-S*X + S*np.sqrt(X**2 + S*4*CV*(CP - CM)))/2
        Q1[where.points['end_valve']] = Q1[where.points['start_valve']]
        H1[where.points['start_valve']] = CP - BP*Q1[where.points['start_valve']]
        H1[where.points['end_valve']] = CM + BM*Q1[where.points['start_valve']]

def run_pump_step(source_head, Q1, H1, Cp, Bp, Cm, Bm, a1, a2, Hs, setting, where):
    if len(where.points['single_pump']) > 0:
        CP = source_head[where.points['single_pump',]]
        BP = np.zeros(len(where.points['single_pump']))
        CM = Cm[where.points['single_pump']]
        BM = Bm[where.points['single_pump']]

        alpha = setting[where.points['single_pump',]]
        A = a2[where.points['single_pump',]]
        B = a1[where.points['single_pump',]]*alpha - BM - BP
        C = Hs[where.points['single_pump',]]*alpha**2 - CM + CP

        root = B ** 2 - 4 * A * C; root[root < 0] = 0
        Q = (-B - np.sqrt(root)) / (2*A); Q[Q < 0] = 0
        Q1[where.points['single_pump']] = Q

        hp = a2[where.points['single_pump',]] * Q**2 + a1[where.points['single_pump',]]*alpha * Q + \
            Hs[where.points['single_pump',]]*alpha**2

        H1[where.points['single_pump']] =  CP + hp

    if len(where.points['start_pump']) > 0:
        CP = Cp[where.points['start_pump']]
        BP = Bp[where.points['start_pump']]
        CM = Cm[where.points['end_pump']]
        BM = Bm[where.points['end_pump']]

        alpha = setting[where.points['start_pump',]]
        A = a2[where.points['start_pump',]]
        B = a1[where.points['start_pump',]]*alpha - BM - BP
        C = Hs[where.points['start_pump',]]*alpha**2 - CM + CP

        root = B ** 2 - 4 * A * C; root[root < 0] = 0
        Q = (-B - np.sqrt(root)) / (2*A); Q[Q < 0] = 0
        Q1[where.points['start_pump']] = Q
        Q1[where.points['end_pump']] = Q

        hp = a2[where.points['start_pump',]] * Q**2 + a1[where.points['start_pump',]]*alpha * Q + \
            Hs[where.points['start_pump',]]*alpha**2

        H1[where.points['start_pump']] =  CP - BP * Q
        H1[where.points['end_pump']] =  H1[where.points['start_pump']] + hp
        H1[where.points['end_pump']] =  CM + BM * Q1[where.points['end_pump']]

def tflow(QT1, QT0, CP, CM, BM, BP, HT0, tau, aT, VA0, C):
    CC = CP + CM # Cp/Bp + Cm/Bm
    BB = BM + BP # 1/Bp + 1/Bm
    Hb = 10.3 # Barometric pressure
    k = 0 # air-chamber orifice head loss coefficient
    m = 1.2 # gas exponent
    return ((CC - QT1)/BB + Hb - (HT0 + tau*(QT1+QT0)/(2*aT))-k*QT1*abs(QT1))* \
        (VA0 - aT*((QT1+QT0)*tau/(2*aT)))**m - C

def tflow_prime(QT1, QT0, CP, CM, BM, BP, HT0, tau, aT, VA0, C):
    CC = CP + CM # Cp/Bp + Cm/Bm
    BB = BM + BP # 1/Bp + 1/Bm
    Hb = 10.3 # Barometric pressure
    k = 0 # air-chamber orifice head loss coefficient
    m = 1.2 # gas exponent
    p1 = (-m*aT/(2 * aT/tau) * (VA0 - (QT0+QT1)*aT/(2 * aT/tau))**(m-1)* \
            ((CC-QT1)/BB + Hb - HT0 - (QT0+QT1)/(2 * aT/tau) - k*QT1*np.abs(QT1)))
    p2 = (-1/BB -1/(2 * aT/tau) - k*2.*QT1*np.sign(QT1)) * \
            (VA0 - (QT0+QT1)*aT/(2 * aT/tau))**m
    return p1+p2

def run_open_protections(H0, H1, Q1, QT, aT, Cp, Bp, Cm, Bm, tau, where):

    CP = Cp[where.points['start_open_protection']]
    BP = Bp[where.points['start_open_protection']]
    CM = Cm[where.points['end_open_protection']]
    BM = Bm[where.points['end_open_protection']]
    CC = CP + CM # Cp/Bp + Cm/Bm
    BB = BM + BP # 1/Bp + 1/Bm

    H1[where.points['start_open_protection']] = \
        (CC + QT + (2*aT*H0[where.points['start_open_protection']]/tau)) / (BB + (2*aT/tau))
    H1[where.points['end_open_protection']] = H1[where.points['start_open_protection']]
    Q1[where.points['start_open_protection']] = CP - H1[where.points['start_open_protection']]*BP
    Q1[where.points['end_open_protection']] = H1[where.points['end_open_protection']]*BM - CM
    QT = Q1[where.points['start_open_protection']] - Q1[where.points['end_open_protection']]

def run_closed_protections(H0, H1, Q1, QT0, QT1, HT0, HT1, VA, htank, aT, Cp, Bp, Cm, Bm, tau, C, where):

    CP = Cp[where.points['start_closed_protection']]
    BP = Bp[where.points['start_closed_protection']]
    CM = Cm[where.points['end_closed_protection']]
    BM = Bm[where.points['end_closed_protection']]
    CC = CP + CM # Cp/Bp + Cm/Bm
    BB = BM + BP # 1/Bp + 1/Bm

    QT1 = optimize.newton(
            tflow, QT0, fprime=tflow_prime,
        args =
            ( QT0, CP, CM, BM, BP, HT0, tau, aT, VA, C), tol=1e-10)

    HT1 = HT0 + (QT0+QT1)/(2 * aT/tau)
    H1[where.points['start_closed_protection']] = (CC - QT1)/BB
    H1[where.points['end_closed_protection']] = H1[where.points['start_closed_protection']]
    Q1[where.points['start_closed_protection']] = CP - H1[where.points['start_closed_protection']]*BP
    Q1[where.points['end_closed_protection']] = H1[where.points['end_closed_protection']]*BM - CM
    VA = (htank - HT1) * aT
    HT0 = HT1
    QT0 = QT1
