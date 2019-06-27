'''

H: 3D Array (pipe, junction, time)
      t=1  t=2  t=3
H = [[0.02 0.02 0.03], p1
     [0.02 0.02 0.03], p2
     [0.02 0.02 0.03], p3
     [0.02 0.02 0.03], p4
     [0.02 0.02 0.03]] p5
'''

import wntr
import pandas as pd
import matplotlib.pyplot as plt

from math import pi

class Simulation:
    def __init__(self, inp_file):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = self.sim.run_sim()
        self.network = self.wn.get_graph()

'''
EPANET NETWORK TABLE
ID | N1 | N2 | Initial flow | N1_head | N2_head | L | f | a | d | n_pieces | TYPE_1 | TYPE_2
'''

#      
sim = Simulation('MOC.inp')
sim.results.link['flowrate']
initial_info = pd.DataFrame(columns = [
     'id', 'n1', 'n2', 'initial_flow', 'n1_head', 'n2_head',
     'length', 'f', 'a', 'd', 'n_pieces', 'dx', 'TYPE_1', 'TYPE2'
     ])

dt = 0.1

for idx, pipe in enumerate(sim.network.edges):
     ID = pipe[2]
     N1 = pipe[0]
     N2 = pipe[1]
     init_flow = float(sim.results.link['flowrate'][ID])
     N1_head = float(sim.results.node['head'][N1])
     N2_head = float(sim.results.node['head'][N2])
     length = sim.results.link['frictionfact'][ID]
     length = float(sim.wn.query_link_attribute('length')[ID])
     f = float(sim.results.link['frictionfact'][ID])
     a = 2000
     d = float(sim.wn.query_link_attribute('diameter')[ID])
     n_pieces = int(length/(dt*a))
     initial_info.loc[idx] = pd.Series({
          'id' : ID,
          'n1' : N1,
          'n2' : N2,
          'initial_flow' : init_flow,
          'n1_head' : N1_head,
          'n2_head' : N2_head,
          'length' : length,
          'f' : f,
          'a' : a,
          'd' : d,
          'n_pieces' : n_pieces, 
          'dx' : length/n_pieces})

nodes = sim.network.nodes

T = 1 # simulation time [s]
pipes = initial_info['id']
H = [[[0 for i in range(int(T/dt))] for j in range(n_pieces+1)] for n_pieces in initial_info['n_pieces']]
Q = [[[0 for i in range(int(T/dt))] for j in range(n_pieces+1)] for n_pieces in initial_info['n_pieces']]

def run_junction_bc(u_pipes, d_pipes, t):
     sc = 0
     sb = 0
     Cp = [0 for i in range(len(u_pipes))]
     Bp = [0 for i in range(len(u_pipes))]
     uQQ = [0 for i in range(len(u_pipes))]
     Cm = [0 for i in range(len(d_pipes))]
     Bm = [0 for i in range(len(d_pipes))]
     dQQ = [0 for i in range(len(d_pipes))]

     for i, p in enumerate(u_pipes['pipes']):
          A = pi*u_pipes['d'][i]**2/4
          g = 9.81
          B = u_pipes['a'][i]/(g*A)
          R = u_pipes['f'][i]*u_pipes['dx'][i]/(2*g*u_pipes['d'][i]*A**2)
          H1 = u_pipes['H1'][i]
          Q1 = u_pipes['Q1'][i]
          Cp[i] = H1 + B*Q1
          Bp[i] = B + R*abs(Q1)
          sc += Cp[i]/Bp[i]
          sb += 1/Bp[i]
     for i, p in enumerate(d_pipes['pipes']):
          A = pi*d_pipes['d'][i]**2/4
          g = 9.81
          B = d_pipes['a'][i]/(g*A)
          R = d_pipes['f'][i]*d_pipes['dx'][i]/(2*g*d_pipes['d'][i]*A**2)
          H1 = d_pipes['H1'][i]
          Q1 = d_pipes['Q1'][i]
          Cm[i] = H1 - B*Q1
          Bm[i] = B + R*abs(Q1)
          sc += Cm[i]/Bm[i]
          sb += 1/Bm[i]
     HH = sc/sb
     for i, p in enumerate(u_pipes['pipes']):
          uQQ[i] = (Cp[i] - HH)/Bp[i]
     for i, p in enumerate(d_pipes['pipes']):
          dQQ[i] = (HH - Cm[i])/Bm[i]

     return (HH, uQQ, dQQ)

def run_demand_bc(H1, Q1, a, d, f, dx):
     A = pi*d**2/4
     g = 9.81
     B = a/(g*A)
     R = f*dx/(2*g*d*A**2)
     QQ = Q1
      # c+ characteristic, i.e., at the end of the pipe
     Cp = H1 + B*Q1
     Bp = B + R*abs(Q1)
     HH = Cp - Bp*Q1
     return (HH, QQ)

def run_valve_bc(H1, Q1, H0, Q0, tau, a, d, f, dx):
     '''
     it is assumed that the valve is at the end of the pipe
     and discharges to the atmosphere
     tau: valve setting (%)
     '''
     A = pi*d**2/4
     g = 9.81
     B = a/(g*A)
     R = f*dx/(2*g*d*A**2)
     Cv = (Q0*tau)**2/(2*H0)
     Cp = H1 + B*Q1
     Bp = B + R*abs(Q1)
     QQ = -Bp*Cv + ((Bp*Cv)**2 + 2*Cv*Cp)**0.5
     HH = Cp - Bp*QQ
     return HH, QQ

def run_source_bc(H0, H1, Q1, a, d, f, dx, type = 'cp'):
     A = pi*d**2/4
     g = 9.81
     B = a/(g*A)
     R = f*dx/(2*g*d*A**2)
     HH = H0
     QQ = 0
     if type == 'cp': # c+ characteristic, i.e., at the end of the pipe
          Cp = H1 + B*Q1
          Bp = B + R*abs(Q1)
          QQ = -(H1 - Cp)/Bp
     elif type == 'cm': # c- characteristic, i.e., at the beggining of the pipe
          Cm = H1 - B*Q1
          Bm = B + R*abs(Q1)
          QQ = (H1 - Cm)/Bm
     return (HH, QQ)

def run_interior_step(Q1, Q2, H1, H2, a, d, f, dx):
     A = pi*d**2/4
     g = 9.81
     B = a/(g*A)
     R = f*dx/(2*g*d*A**2)
     Cp = H1 + B*Q1
     Cm = H2 - B*Q2
     Bp = B + R*abs(Q1)
     Bm = B + R*abs(Q2)
     HH = (Cp*Bm + Cm*Bp)/(Bp + Bm)
     QQ = (Cp - Cm)/(Bp + Bm) 
     return (HH, QQ)

#### SIM STARTS !!!

for i, p in enumerate(pipes):
     Np = int(initial_info['n_pieces'][i])
     H0 = float(initial_info['n1_head'][i])
     dH0 = float(initial_info['n1_head'][i]) - float(initial_info['n2_head'][i])
     Q0 = float(initial_info['initial_flow'][i])
     for j in range(Np+1):
          H[i][j][0] = H0 - j*dH0/Np
          Q[i][j][0] = Q0

st = [max(0, 1-0.2*i) for i in range(int(T/dt))]

for t in range(1,int(T/dt)):
     for i, p in enumerate(pipes):
          Np = int(initial_info['n_pieces'][i])
          for j in range(1,Np):
               H1 = H[i][j-1][t-1]
               Q1 = Q[i][j-1][t-1]
               H2 = H[i][j+1][t-1]
               Q2 = Q[i][j+1][t-1]
               
               HH, QQ = run_interior_step(
                    Q1, Q2, H1, H2,
                    float(initial_info['a'][i]),
                    float(initial_info['d'][i]), 
                    float(initial_info['f'][i]),
                    float(initial_info['dx'][i]))
               H[i][j][t] = HH
               Q[i][j][t] = QQ

          # Boundary conditions
          if p == 'P1':
               HH, QQ = run_source_bc(
                    H[i][0][t-1], H[i][1][t-1], Q[i][1][t-1],
                    float(initial_info['a'][i]),
                    float(initial_info['d'][i]), 
                    float(initial_info['f'][i]),
                    float(initial_info['dx'][i]),
                    'cm')
               H[i][0][t] = HH
               Q[i][0][t] = QQ
          if p == 'P2':
               HH, QQ = run_valve_bc(
                    H[i][-2][t-1], Q[i][-2][t-1],
                    H[i][j][0], Q[i][j][0],
                    st[t],
                    float(initial_info['a'][i]),
                    float(initial_info['d'][i]), 
                    float(initial_info['f'][i]),
                    float(initial_info['dx'][i]))
               H[i][-1][t] = HH
               Q[i][-1][t] = QQ
          if p == 'P3':
               HH, QQ = run_demand_bc(
                    H[i][-2][t-1], Q[i][-2][t-1],
                    float(initial_info['a'][i]),
                    float(initial_info['d'][i]), 
                    float(initial_info['f'][i]),
                    float(initial_info['dx'][i]))
               H[i][-1][t] = HH
               Q[i][-1][t] = QQ
     # Junction BC
     u_pipes = [2]
     d_pipes = [0, 1]
     HH, uQQ, dQQ = run_junction_bc(
          u_pipes = { 
               'pipes' : u_pipes, 
               'a' : list(map(float, initial_info['a'][u_pipes])),
               'f' : list(map(float, initial_info['f'][u_pipes])),
               'dx' : list(map(float, initial_info['dx'][u_pipes])),
               'd' : list(map(float, initial_info['d'][u_pipes])),
               'H1' : [H[i][-2][t-1] for i in u_pipes],
               'Q1' : [Q[i][-2][t-1] for i in u_pipes]
          },
          d_pipes = { 
               'pipes' : d_pipes, 
               'a' : list(map(float, initial_info['a'][d_pipes])),
               'f' : list(map(float, initial_info['f'][d_pipes])),
               'dx' : list(map(float, initial_info['dx'][d_pipes])),
               'd' : list(map(float, initial_info['d'][d_pipes])),
               'H1' : [H[i][1][t-1] for i in d_pipes],
               'Q1' : [Q[i][1][t-1] for i in d_pipes]
          },
          t = t)
     H[2][-1][t] = HH
     H[0][0][t] = HH
     H[1][0][t] = HH
     Q[2][-1][t] = uQQ[0]
     Q[0][0][t] = dQQ[0]
     Q[1][0][t] = dQQ[1]

tt = [dt*i for i in range(int(T/dt))]

for i, p in enumerate(initial_info['id']):
     for j in range(initial_info['n_pieces'][i]):
          plt.plot(tt, H[i][j])
     plt.title('Head(t) in components of pipe %s' % p)
     plt.show()
     for j in range(initial_info['n_pieces'][i]):
          plt.plot(tt, Q[i][j])
     plt.title('Flow(t) in components of pipe %s' % p)
     plt.show()