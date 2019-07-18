from phammer.simulation.sim import Simulation

input_file = 'example_files/LoopedNet.inp'
sim = Simulation(input_file,
    duration = 20, # [s]
    time_step = 0.01, # [s]
    default_wave_speed = 1200)