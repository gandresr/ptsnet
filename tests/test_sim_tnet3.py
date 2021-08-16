import sys
import os
import numpy as np
import matplotlib.pyplot as plt

solver = sys.argv[1]
test = sys.argv[2]
junctions_burst = ['JUNCTION-34', 'JUNCTION-16', 'JUNCTION-20', 'JUNCTION-30', 'JUNCTION-45', 'JUNCTION-90']
junctions_burst = ['JUNCTION-34', 'JUNCTION-16', 'JUNCTION-73', 'JUNCTION-68']
if solver == 'ptsnet':
    from ptsnet.simulation.sim import PTSNETSimulation
    from ptsnet.utils.io import get_example_path
else:
    import tsnet

def test_valve_closure(solver = 'ptsnet'):
    dt = 0.00635

    if solver == 'ptsnet':
        sim = PTSNETSimulation(
            inpfile = get_example_path('tnet3_hammer'),
            settings = {
                'save_results' : True,
                'duration' : 20,
                'time_step' : dt,
                'show_progress' : True,
                'warnings_on' : False
            },
            default_wave_speed = 1200
        )

        print("Time step [PTSNET]: ", sim.settings.time_step)
        valves = sim.ic['valve'].labels
        sim.add_curve('V_BUTTERFLY', 'valve',
            [1, 0.8, 0.6, 0.4, 0.2, 0],
            [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])
        sim.assign_curve_to('V_BUTTERFLY', valves)
        NN = int(1//dt)
        sim.define_valve_settings('VALVE-175', np.linspace(0,1,NN), np.linspace(1,0,NN))
        sim.initialize()
        while not sim.is_over:
            sim.run_step()

        head_change = np.zeros((len(junctions)+1, sim.settings.time_steps))
        head_change[0] = sim['time']
        for i, j in enumerate(junctions):
            head_change[i+1] = sim['node'].head[j]# - sim['node'].head[j][0]
            plt.plot(head_change[0], head_change[i+1], label=j, linewidth=1.5)

        np.savetxt(f'results/valve_{solver}_tnet3.csv', head_change.T, delimiter=',', header=','.join(['TIME'] + junctions))

    elif solver == 'tsnet':
        tm = tsnet.network.TransientModel('/home/watsup/Documents/Github/ptsnet/ptsnet/examples/TNET3_HAMMER.inp')

        # Set wavespeed
        tm.set_wavespeed(1200.)

        # Set time step
        tf = 20 # simulation period [s]
        tm.set_time(tf,dt)
        ts = 0 # valve closure start time [s]
        tc = 1 # valve closure period [s]
        se = 0 # end open percentage [s]
        m = 1 # closure constant [dimensionless]
        tm.valve_closure('VALVE-175',[tc,ts,se,m])

        # Initialize steady state simulation
        tm = tsnet.simulation.Initializer(tm,0)

        # Transient simulation
        tm = tsnet.simulation.MOCSimulator(tm)
        print("Time step [TSNET]: ", tm.time_step)
        head_change = np.zeros((len(junctions)+1, len(tm.simulation_timestamps)))
        head_change[0] = tm.simulation_timestamps
        for i, j in enumerate(junctions):
            head_change[i+1] = tm.nodes[j].head# - sim['node'].head[j][0]
            plt.plot(head_change[0], head_change[i+1], label=j, linewidth=1.5)

        np.savetxt(f'results/valve_{solver}_tnet3.csv', head_change.T, delimiter=',', header=','.join(['TIME'] + junctions))
        plt.xlabel('Time [s]')
        plt.ylabel('Head change [m]')
        plt.xlim(0,20)
        plt.legend()
        plt.show()

def test_pump_shutoff(solver = 'ptsnet'):
    dt = 0.00635

    if solver == 'ptsnet':
        sim = PTSNETSimulation(
            inpfile = get_example_path('tnet3_hammer'),
            settings = {
                'save_results' : True,
                'duration' : 20,
                'time_step' : dt,
                'show_progress' : True,
                'warnings_on' : False
            },
            default_wave_speed = 1200
        )

        valves = sim.ic['valve'].labels
        sim.add_curve('V_BUTTERFLY', 'valve',
            [1, 0.8, 0.6, 0.4, 0.2, 0],
            [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])
        sim.assign_curve_to('V_BUTTERFLY', valves)
        tt = 2; NN = int(tt//dt)
        sim.define_pump_settings('PUMP-172', np.linspace(0,tt,NN), np.linspace(1,0,NN))
        # sim.add_surge_protection('JUNCTION-34', 'open', 1, tank_height=1, water_level=0.2)
        sim.initialize()
        while not sim.is_over:
            sim.run_step()

        head_change = np.zeros((len(junctions)+1, sim.settings.time_steps))
        head_change[0] = sim['time']
        for i, j in enumerate(junctions):
            head_change[i+1] = sim['node'].head[j]# - sim['node'].head[j][0]

        np.savetxt(f'results/pump_{solver}_tnet3.csv', head_change.T, delimiter=',', header=','.join(['TIME'] + junctions))

    elif solver == 'tsnet':
        tm = tsnet.network.TransientModel('/home/watsup/Documents/Github/ptsnet/ptsnet/examples/TNET3_HAMMER.inp')

        # Set wavespeed
        tm.set_wavespeed(1200.)

        # Set time step
        tf = 20 # simulation period [s]
        tm.set_time(tf,dt)
        # Set pump shut off
        tc = 2 # pump closure period
        ts = 0 # pump closure start time
        se = 0 # end open percentage
        m = 1 # closure constant
        tm.pump_shut_off('PUMP-172', [tc,ts,se,m])

        # Initialize steady state simulation
        tm = tsnet.simulation.Initializer(tm,0)

        # Transient simulation
        tm = tsnet.simulation.MOCSimulator(tm)

        head_change = np.zeros((len(junctions)+1, len(tm.simulation_timestamps)))
        head_change[0] = tm.simulation_timestamps
        for i, j in enumerate(junctions):
            head_change[i+1] = tm.nodes[j].head# - sim['node'].head[j][0]
            plt.plot(head_change[0], head_change[i+1], label=j, linewidth=1.5)

        np.savetxt(f'results/pump_{solver}_tnet3.csv', head_change.T, delimiter=',', header=','.join(['TIME'] + junctions))
        plt.xlabel('Time [s]')
        plt.ylabel('Head change [m]')
        plt.xlim(0,20)
        plt.legend()
        plt.show()

def test_burst(solver = 'ptsnet'):
    dt = 0.00635

    if solver == 'ptsnet':
        sim = PTSNETSimulation(
            inpfile = get_example_path('tnet3_hammer'),
            settings = {
                'save_results' : True,
                'duration' : 20,
                'time_step' : dt,
                'show_progress' : True,
                'warnings_on' : False,
            },
            default_wave_speed = 1200
        )

        valves = sim.ic['valve'].labels
        sim.add_curve('V_BUTTERFLY', 'valve',
            [1, 0.8, 0.6, 0.4, 0.2, 0],
            [0.067, 0.044, 0.024, 0.011, 0.004, 0.   ])
        sim.assign_curve_to('V_BUTTERFLY', valves)
        tt = 2; NN = int(tt//dt)
        sim.define_burst_settings('JUNCTION-20', np.linspace(0,tt,NN), np.linspace(0,0.0003,NN))
        sim.initialize()
        while not sim.is_over:
            sim.run_step()

        head_change = np.zeros((len(junctions_burst)+1, sim.settings.time_steps))
        head_change[0] = sim['time']
        for i, j in enumerate(junctions_burst):
            head_change[i+1] = sim['node'].head[j]# - sim['node'].head[j][0]

        np.savetxt(f'results/burst_{solver}_tnet3.csv', head_change.T, delimiter=',', header=','.join(['TIME'] + junctions_burst))

    elif solver == 'tsnet':
        tm = tsnet.network.TransientModel('/home/watsup/Documents/Github/ptsnet/ptsnet/examples/TNET3_HAMMER.inp')

        # Set wavespeed
        tm.set_wavespeed(1200.)

        # Set time step
        tf = 20 # simulation period [s]
        tm.set_time(tf,dt)
        # Add burst
        ts = 0 # burst start time
        tc = 1 # time for burst to fully develop
        final_burst_coeff = 0.0003 # final burst coeff [ m^3/s/(m H20)^(1/2)]
        tm.add_burst('JUNCTION-20', ts, tc, final_burst_coeff)

        # Initialize steady state simulation
        tm = tsnet.simulation.Initializer(tm,0)

        # Transient simulation
        tm = tsnet.simulation.MOCSimulator(tm)

        head_change = np.zeros((len(junctions_burst)+1, len(tm.simulation_timestamps)))
        head_change[0] = tm.simulation_timestamps
        for i, j in enumerate(junctions_burst):
            head_change[i+1] = tm.nodes[j].head# - sim['node'].head[j][0]
            plt.plot(head_change[0], head_change[i+1], label=j, linewidth=1.5)

        np.savetxt(f'results/burst_{solver}_tnet3.csv', head_change.T, delimiter=',', header=','.join(['TIME'] + junctions_burst))
        plt.xlabel('Time [s]')
        plt.ylabel('Head change [m]')
        plt.xlim(0,20)
        plt.legend()
        plt.show()

tests = ['valve', 'pump', 'leak', 'burst']

if solver != 'plot':
    if test == 'valve':
        test_valve_closure(solver)
    elif test == 'pump':
        test_pump_shutoff(solver)
    elif test == 'burst':
        test_burst(solver)
else:
    dptsnet = np.genfromtxt(f'results/{test}_ptsnet_tnet3.csv', dtype=float, delimiter=',', names=True)
    dtsnet = np.genfromtxt(f'results/{test}_tsnet_tnet3.csv', dtype=float, delimiter=',', names=True)
    dhammer = np.genfromtxt(f'results/{test}_hammer_tnet3.csv', dtype=float, delimiter=',', names=True)
    if test == 'burst': junctions = junctions_burst
    for jj in junctions:
        j = jj.replace('-','')
        plt.plot(dptsnet['TIME'], dptsnet[j], label='PTSNET', linewidth=1.5)
        plt.plot(dtsnet['TIME'], dtsnet[j], label='TSNET', linewidth=1.5)
        plt.plot(dhammer['TIME'], dhammer[j], label='HAMMER', linewidth=1.5)
        plt.xlabel('Time [s]')
        plt.ylabel('Head [m]')
        plt.title(f'Head at {j}')
        plt.xlim(0,20)
        # plt.ylim(260,270)
        plt.legend()
        plt.show()