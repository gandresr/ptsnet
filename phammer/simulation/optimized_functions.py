from numba import njit

PARALLEL = True

@njit(parallel = PARALLEL)
def run_interior_step(Q1, Q2, H1, H2, B, R):
    """Solves flow and head for interior points

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
        Q1 {array} -- initial flow
        Q2 {array} -- flow solution for the next time step
        H1 {array} -- initial head
        H2 {array} -- head solution for the next time step
        B {array} -- coefficients B[i] = a/(g*A)
        R {array} -- coefficients R[i] = f*dx/(2*g*D*A**2)
    """

    for i in range(1, len(H1)-1):
        # The first and last nodes are skipped in the  loop considering
        # that they are boundary nodes (every interior node requires an
        # upstream and a downstream neighbor)
        H2[i] = ((H1[i-1] + B[i]*Q1[i-1])*(B[i] + R[i]*abs(Q1[i+1])) \
            + (H1[i+1] - B[i]*Q1[i+1])*(B[i] + R[i]*abs(Q1[i-1]))) \
            / ((B[i] + R[i]*abs(Q1[i-1])) + (B[i] + R[i]*abs(Q1[i+1])))
        Q2[i] = ((H1[i-1] + B[i]*Q1[i-1]) - (H1[i+1] - B[i]*Q1[i+1])) \
            / ((B[i] + R[i]*abs(Q1[i-1])) + (B[i] + R[i]*abs(Q1[i+1])))

@njit(parallel = PARALLEL)
def run_junction_step(
    junctions_int, Q1, H1, Q2, H2, B, R, junction_type,
    demand, emitter_coeff, emitter_setting,
    RESERVOIR, PIPE, EMITTER, N, DOWN, UP):
    """Solves flow and head for boundary nodes in junctions

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    Arguments:
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
        emitter_setting {array} -- setting value emitter \in [0, 1]
        RESERVOIR {int} -- constant for junctions of type 'reservoir'
        PIPE {int} -- constant for junctions of type 'pipe'
        EMITTER {int} -- constant for junctions of type 'emitter'
        UP {int} -- row index in junctions_int to extract upstream_neighbors_num
        DOWN {int} -- row index in junctions_int to extract downstream_neighbors_num
        N {int} -- row index to extract the first downstream node in table junctions_int
    """
    for j_id in range(junctions_int.shape[1]):

        downstream_num = junctions_int[DOWN, j_id]
        upstream_num = junctions_int[UP, j_id]

        # Junction is a reservoir
        if junction_type[j_id] == RESERVOIR:
            for j in range(downstream_num):
                k = junctions_int[j+N, j_id]
                H2[k] = H1[k]
                Q2[k] = (H1[k] - H1[k+1] + B[k+1]*Q1[k+1]) \
                        / (B[k+1] + R[k+1]*abs(Q1[k+1]))
            for j in range(downstream_num, upstream_num+downstream_num):
                k = junctions_int[j+N, j_id]
                H2[k] = H1[k]
                Q2[k] = (H1[k-1] + B[k-1]*Q1[k-1] - H1[k]) \
                        / (B[k-1] + R[k-1]*abs(Q1[k-1]))
        elif junction_type[j_id] == EMITTER or junction_type[j_id] == PIPE:
            sc = 0
            sb = 0
            Cm = [0 for i in range(downstream_num)]
            Bm = [0 for i in range(downstream_num)]
            Cp = [0 for i in range(upstream_num)]
            Bp = [0 for i in range(upstream_num)]

            for j in range(downstream_num): # C-
                k = junctions_int[j+N, j_id]
                Cm[j] = H1[k+1] - B[k+1]*Q1[k+1]
                Bm[j] = B[k+1] + R[k+1]*abs(Q1[k+1])
                sc += Cm[j] / Bm[j]
                sb += 1 / Bm[j]

            for j in range(downstream_num, upstream_num+downstream_num): # C+
                k = junctions_int[j+N, j_id]
                Cp[j] = H1[k-1] + B[k-1]*Q1[k-1]
                Bp[j] = B[k-1] + R[k-1]*abs(Q1[k-1])
                sc += Cp[j] / Bp[j]
                sb += 1 / Bp[j]

            Z = sc/sb + demand[j_id]/sb
            HH = Z

            if junction_type[j_id] == EMITTER:
                K = (emitter_setting[j_id]*emitter_coeff[j_id]/sb)**2
                HH = ((2*Z + K) - (K**2 + 4*Z*K)**0.5) / 2

            for j in range(downstream_num): # C-
                k = junctions_int[j+N, j_id]
                H2[k] = HH
                Q2[k] = (HH - Cm[j]) / Bm[j]

            for j in range(downstream_num, upstream_num+downstream_num): # C+
                k = junctions_int[j+N, j_id]
                H2[k] = HH
                Q2[k] = (Cp[j] - HH) / Bp[j]