import matplotlib.pyplot as plt

from phammer.mesh.mesh import Mesh
from phammer.simulation.initial_conditions import get_initial_conditions

input_file = '../examples/files/LoopedNet_1.inp'
time_step = 0.01
wave_speed = 1200

mesh = Mesh(
    input_file,
    time_step,
    default_wave_speed = wave_speed
)

Q0, H0 = get_initial_conditions(mesh)
plt.plot(H0)
plt.show()