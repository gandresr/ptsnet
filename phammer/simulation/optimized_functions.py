from numba import njit

PARALLEL = True

@njit(parallel = PARALLEL)
def run_interior_step(t, Q, H, B, R):
    """Solves flow and head for interior points

    All the numpy arrays are passed by reference,
    therefore, it is possible to override the
    values of the parameters of the function

    TODO: UPDATE ARGUMENTS
    """

    for i in range(1, H.shape[1]-1):
        # The first and last nodes are skipped in the  loop considering
        # that they are boundary nodes (every interior node requires an
        # upstream and a downstream neighbor)
        H[t][i] = ((H[t-1][i-1] + B[i]*Q[t-1][i-1])*(B[i] + R[i]*abs(Q[t-1][i+1])) \
            + (H[t-1][i+1] - B[i]*Q[t-1][i+1])*(B[i] + R[i]*abs(Q[t-1][i-1]))) \
            / ((B[i] + R[i]*abs(Q[t-1][i-1])) + (B[i] + R[i]*abs(Q[t-1][i+1])))
        Q[t][i] = ((H[t-1][i-1] + B[i]*Q[t-1][i-1]) - (H[t-1][i+1] - B[i]*Q[t-1][i+1])) \
            / ((B[i] + R[i]*abs(Q[t-1][i-1])) + (B[i] + R[i]*abs(Q[t-1][i+1])))

@njit(parallel = PARALLEL)
def run_junction_step(Q, H, D, E, B, R, nodes_int, nodes_float, nodes_obj, RESERVOIR, PIPE, EMITTER):
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
        emitter_setting {array} -- setting value emitter \in [0, 1]
        RESERVOIR {int} -- constant for junctions of type 'reservoir'
        PIPE {int} -- constant for junctions of type 'pipe'
        EMITTER {int} -- constant for junctions of type 'emitter'
        UP {int} -- row index in junctions_int to extract upstream_neighbors_num
        DOWN {int} -- row index in junctions_int to extract downstream_neighbors_num
        N {int} -- row index to extract the first downstream node in table junctions_int
    """
    for node_id in range(nodes.shape[1]):

        dpoints = nodes_obj.downstream_points[node_id]
        upoints = nodes_obj.upstream_points[node_id]
        node_type = nodes_int.node_type[nodes_id]

        # Junction is a reservoir
        if node_type == RESERVOIR:
            for k in dpoints:
                H[t][k] = H[t-1][k]
                Q[t][k] = (H[t-1][k] - H[t-1][k+1] + B[k+1]*Q[t-1][k+1]) \
                        / (B[k+1] + R[k+1]*abs(Q[t-1][k+1]))
            for k in upoints:
                H[t][k] = H[t-1][k]
                Q[t][k] = (H[t-1][k-1] + B[k-1]*Q[t-1][k-1] - H[t-1][k]) \
                        / (B[k-1] + R[k-1]*abs(Q[t-1][k-1]))
        elif node_type == EMITTER or node_type == PIPE:
            sc = 0
            sb = 0
            Cm = [0 for i in range(len(dpoints))]
            Bm = [0 for i in range(len(dpoints))]
            Cp = [0 for i in range(len(upoints))]
            Bp = [0 for i in range(len(upoints))]

            for j, k in enumerate(dpoints): # C-
                Cm[j] = H[t-1][k+1] - B[k+1]*Q[t-1][k+1]
                Bm[j] = B[k+1] + R[k+1]*abs(Q[t-1][k+1])
                sc += Cm[j] / Bm[j]
                sb += 1 / Bm[j]

            for j, k in enumerate(upoints): # C+
                Cp[j] = H[t-1][k-1] + B[k-1]*Q[t-1][k-1]
                Bp[j] = B[k-1] + R[k-1]*abs(Q[t-1][k-1])
                sc += Cp[j] / Bp[j]
                sb += 1 / Bp[j]

            Z = sc/sb
            Ke = nodes_float.emitter_setting[node_id]*nodes_float.emitter_coeff[node_id]
            Kd = nodes_float.demand_coeff[node_id]
            K = ((Ke+Kd)/sb)**2
            HH = ((2*Z + K) - (K**2 + 4*Z*K)**0.5) / 2

            E[t][node_id] = Ke*(2*G*HH)**0.5
            D[t][node_id] = Kd*(2*G*HH)**0.5

            for k in dpoints: # C-
                H[t][k] = HH
                Q[t][k] = (HH - Cm[j]) / Bm[j]

            for k in upoints: # C+
                H[t][k] = HH
                Q[t][k] = (Cp[j] - HH) / Bp[j]