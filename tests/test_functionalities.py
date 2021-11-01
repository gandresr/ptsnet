


sim = PTSNETSimulation(inpfile = get_example_path('B0_SURGE'))
sim.define_valve_operation(sim.all_valves, initial_setting=1, final_setting=0)
sim.add_surge_protection('N7', 'closed', 1000, 10, 1)
sim.add_burst('N5', 5.13, 0, 1)
sim.run()

plt.clf()
plt.plot(sim['time'], sim['node'].head['NV-A'], label='NV-A')
plt.plot(sim['time'], sim['node'].head['N7'], label='N7')
plt.plot(sim['time'], sim['node'].head['N5'], label='N5')
plt.plot(sim['time'], sim['node'].head['N6'], label='N6')
plt.xlabel('Time [s]')
plt.ylabel('Head [m]')
plt.legend()
plt.savefig('fig.png')

def test_closed_surge_tank():
    sim = PTSNETSimulation(inpfile = get_example_path('pipe_in_series'))
    sim.define_valve_operation(sim.all_valves, start_time=0, end_time=1, initial_setting=1, final_setting=0)
    sim.add_surge_protection('N1', 'open', 0.5)
    sim.run()

    plt.plot(sim['time'], sim['node'].head['N0'], label='N0')
    plt.plot(sim['time'], sim['node'].head['N1'], label='N1')
    plt.plot(sim['time'], sim['node'].head['N2'], label='N2')
    plt.xlabel('Time [s]')
    plt.ylabel('Head [m]')
    plt.legend()
    plt.savefig('fig.png')

test_open_surge_tank()