import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wntr
import tsnet

from phammer.simulation.sim import HammerSimulation
from time import time

default_time_step = 0.01
duration = 50

inpfiles = (
    '/home/griano/Documents/Github/phammer/example_files/simple_pump.inp',
    '/home/griano/Documents/Github/phammer/example_files/Tnet1.inp',
    '/home/griano/Documents/Github/phammer/example_files/Tnet2.inp')

for i, inpfile in enumerate(inpfiles):
    tm = tsnet.network.TransientModel(inpfile)
    tm.set_wavespeed(1200.) # m/s
    tm.set_time(duration)

    if i == 0: # simple_pump.py
        pump_op = [5,0,0,1]
        tm.pump_shut_off('pump', pump_op)
    elif i == 1: # TNet1.py
        valve_op = [2,0,0,1]
        tm.valve_closure('VALVE', valve_op)
    elif i == 2: # TNet2.py
        pump_op = [1,0,0,1]
        tm.pump_shut_off('PUMP2', pump_op)

    # Run TSNet simulation
    tm = tsnet.simulation.Initializer(tm, 0, 'DD')
    tm = tsnet.simulation.MOCSimulator(tm)
    time_step = tm.time_step

    sim = HammerSimulation(
    inpfile,
    {
        'time_step' : default_time_step,
        'duration' : duration,
        'skip_compatibility_check' : False,
    },
    period = 0,
    default_wave_speed = 1200)

    if i > 0:
        sim.add_curve('V_BUTTERFLY', 'valve',
            [1. , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0. ],
            [5.0, 2.5, 1.25, 0.625, 0.333, 0.17, 0.1, 0.0556, 0.0313, 0.0167, 0.0])

        valves = sim.wn.valve_name_list
        sim.assign_curve_to('V_BUTTERFLY', valves)

    if i == 0: # simple_pump.py
        sim.define_pump_settings('pump', np.linspace(0, 5, int(5/sim.settings.time_step)), np.linspace(1, 0, int(5/sim.settings.time_step)))
    elif i == 1: # TNet1.py
        sim.define_valve_settings('VALVE', np.linspace(0, 2, int(2/sim.settings.time_step)), np.linspace(1, 0, int(2/sim.settings.time_step)))
    elif i == 2: # TNet2.py
        sim.define_pump_settings('PUMP2', np.linspace(0, 1, int(1/sim.settings.time_step)), np.linspace(1, 0, int(1/sim.settings.time_step)))

    sim.initialize()
    while not sim.is_over:
        sim.run_step()

    tt = np.linspace(0, duration, sim.settings.time_steps)
    # Compare Results
    if i == 0: # simple_pump.py
        node = tm.get_node('2')
        plt.plot(tm.simulation_timestamps, node.head, label='TSNet')
        plt.plot(tt, sim.worker.node_results.head['2'], label='phammer')
        plt.xlim([tm.simulation_timestamps[0],tm.simulation_timestamps[-1]])
        plt.title('Pressure Head at Node 2')
        plt.xlabel("Time [s]")
        plt.ylabel("Pressure Head [m]")
        plt.legend(loc='best')
        plt.show()
        pipe = tm.get_link('p2')
        plt.plot(tm.simulation_timestamps, pipe.start_node_flowrate, label='TSNet')
        plt.plot(tt, sim.worker.pipe_start_results.flowrate['p2'], label='phammer')
        plt.xlim([tm.simulation_timestamps[0],tm.simulation_timestamps[-1]])
        plt.title('Flowrate at pipe p2')
        plt.xlabel("Time [s]")
        plt.ylabel("Flow [$m/s^2$]")
        plt.legend(loc='best')
        plt.show()
    elif i == 1:
        node = tm.get_node('N3')
        plt.plot(tm.simulation_timestamps, node.head, label='TSNet')
        plt.plot(tt, sim.worker.node_results.head['N3'], label='phammer')
        plt.xlim([tm.simulation_timestamps[0],tm.simulation_timestamps[-1]])
        plt.title('Pressure Head at Node N3')
        plt.xlabel("Time [s]")
        plt.ylabel("Pressure Head [m]")
        plt.legend(loc='best')
        plt.show()
        pipe = tm.get_link('P7')
        plt.plot(tm.simulation_timestamps, pipe.end_node_flowrate, label='TSNet')
        plt.plot(tt, sim.worker.pipe_end_results.flowrate['P7'], label='phammer')
        plt.xlim([tm.simulation_timestamps[0],tm.simulation_timestamps[-1]])
        plt.title('Flowrate at pipe P7')
        plt.xlabel("Time [s]")
        plt.ylabel("Flow [$m/s^2$]")
        plt.legend(loc='best')
        plt.show()
    elif i == 2:
        node = tm.get_node('JUNCTION-105')
        plt.plot(tm.simulation_timestamps, node.head, label='TSNet')
        plt.plot(tt, sim.worker.node_results.head['JUNCTION-105'], label='phammer')
        plt.xlim([tm.simulation_timestamps[0],tm.simulation_timestamps[-1]])
        plt.title('Pressure Head at Node JUNCTION-105')
        plt.xlabel("Time [s]")
        plt.ylabel("Pressure Head [m]")
        plt.legend(loc='best')
        plt.show()
        pipe = tm.get_link('PIPE-109')
        plt.plot(tm.simulation_timestamps, pipe.end_node_flowrate, label='TSNet')
        plt.plot(tt, sim.worker.pipe_end_results.flowrate['PIPE-109'], label='phammer')
        plt.xlim([tm.simulation_timestamps[0],tm.simulation_timestamps[-1]])
        plt.title('Flowrate at pipe PIPE-109')
        plt.xlabel("Time [s]")
        plt.ylabel("Flow [$m/s^2$]")
        plt.legend(loc='best')
        plt.show()