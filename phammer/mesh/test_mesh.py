from mesh import Mesh
from os import getcwd, sep

Mesh(getcwd() + sep + 'LoopedNet_1.inp', 0.01, default_wave_speed = 1200)