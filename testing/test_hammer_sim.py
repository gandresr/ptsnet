from phammer.simulation.sim import HammerSimulation

inpfile = '/home/watsup/Documents/Github/hammer-net/example_files/LoopedNet.inp'

sim = HammerSimulation(inpfile, {
    'time_step' : 0.01
})