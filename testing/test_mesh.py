from phammer.mesh.mesh import Mesh
from os import getcwd, sep


input_file = '../examples/files/LoopedNet_1.inp'
time_step = 0.01
wave_speed = 1200

mesh = Mesh(
    input_file,
    time_step,
    default_wave_speed = wave_speed
)