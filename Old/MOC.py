'''

H: 3D Array (pipe, junction, time)
      t=1  t=2  t=3
H = [[0.02 0.02 0.03], p1
     [0.02 0.02 0.03], p2
     [0.02 0.02 0.03], p3
     [0.02 0.02 0.03], p4
     [0.02 0.02 0.03]] p5
'''


def run_interior_step(Q1, Q2, H1, H2):
    B = a/(g*A)
    R = f*dx/(2*g*D*A**2)
    Cp = H1 + B*Q1
    Cm = H2 - B*Q2
    Bp = B + R*abs(Q1)
    Bm = B + R*abs(Q2)
    H = (Cp*Bm + Cm*Bp)/(Bp + Bm)
    Q = (Cp - Cm)/(Bp + Bm)    
    return (Q, H)

def solve_bc(n, ):