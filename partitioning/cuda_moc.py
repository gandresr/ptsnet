import numpy as np
from pmoc import MOC_network as Net
from pmoc import MOC_simulation as Sim
from pmoc import Wall_clock as WC
from time import time
from pprint import pprint

# CUDA
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

point_kernel = SourceModule('''
    __global__ void point_step(
        float * HH,
        float * QQ,
        float * Q1_point, 
        float * Q2_point, 
        float * H1_point, 
        float * H2_point, 
        float * wavespeed_point,
        float * D_point,
        float * frictionfact_point,
        float * dx_point,
        float * A_point)
    {
        const int i = threadIdx.x;
        float B, R, Cp, Cm, Bp, Bm;

        B = wavespeed[i]/(9.81*A_point[i]);
        R = frictionfact[i]*dx_point[i]/(2*9.81*D_point[i]*A_point[i]*A_point[i]);
        Cp = H1_point[i] + B*Q1_point[i];
        Cm = H2_point[i] - B*Q2_point[i];
        Bp = B + R*abs(Q1_point[i]);
        Bm = B + R*abs(Q2_point[i]);
        HH_points[i] = (Cp*Bm + Cm*Bp)/(Bp + Bm);
        QQ_points[i] = (Cp - Cm)/(Bp + Bm);
    }
''')

valve_kernel = SourceModule('''
    __global__ void valve_step(
        float * H1_valve, 
        float * Q1_valve, 
        float * H0_valve, 
        float * Q0_valve, 
        float * setting, 
        float * wavespeed_valve, 
        float * D_valve, 
        float * frictionfact_valve, 
        float * dx_valve,
        float * area_valve
    )
        float B, R, Cv, Cp, Bp
        B = wavespeed_valve[i]/(9.81*area_valve[i])
        R = f*dx/(2*g*d*area_valve*area_valve)
        Cv = (Q0*tau)**2/(2*H0)
        Cp = H1 + B*Q1
        Bp = B + R*abs(Q1)
        QQ = -Bp*Cv + ((Bp*Cv)**2 + 2*Cv*Cp)**0.5
        HH = Cp - Bp*QQ
        return HH, QQ
''')

clk = WC()
clk.tic()
network = Net("models/LoopedNet.inp")
network.define_wavespeeds(default_wavespeed = 1200)
network.define_segments(0.1)
network.define_mesh()
# network.write_mesh()
# network.define_partitions(2)
clk.toc()

point_step = point_kernel.get_function("point_step")
HH = np.zeros()
QQ,
Q1_point, 
Q2_point, 
H1_point, 
H2_point, 
wavespeed_point,
D_point,
frictionfact_point,
dx_point,
A_point