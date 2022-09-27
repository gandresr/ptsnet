## Publication Results

In this folder you will find four Jupyter Notebooks, which contain test cases for PTSNet's functionalities.

The file `1_simulate_scenarios.ipynb` demonstrates how to import PTSNet, define simulation settings, generate a transient model, set up a transient event, run a transient simulation, and get simulation results.

The file `2_get_results.ipynb` demonstrates how to load results saved in previous simulations (without the need to rerun the simulations).

The file `3_analytics.ipynb` contains examples of analytics functionalities to estimate simulation times, number of processors, and wave speed error for a specific simulation.

The file `4_SI_figures.ipynb` exeutes functions to compare PTSNet simulation results with respect to TSNet and Hammer v8i.

The folder *HAMMER* contains Hammer v8i files that were used to compare simulation results for valve closure, pump shut-off, and burst simulations on the TNET3 example.

The folder *SI_results* contains summarized results to compare transient simulations that were executed with Hammer v8i, TSNET, and PTSNET.

Below you can find a comprehensive list of all the properties and results that can be extracted from the model. All the numerical variables representing a physical property of the object are expressed in SI units.

**Node results**: can be extracted through `PTSNETSimulation['node'].<result_name>`
- 'head'
- 'leak_flow': leaking flowrate
- 'demand_flow': demand flowrate

**Pipe start results**: can be extracted through `PTSNETSimulation['pipe.start'].<result_name>`
- 'flowrate'

**Pipe end results**: can be extracted through `PTSNETSimulation['pipe.end'].<result_name>`
- 'flowrate'

**Closed protection results**: can be extracted through `PTSNETSimulation['closed_protection'].<result_name>`
- 'water_level': level of water inside of the surge tank

**Node properties**: can be extracted through `PTSNETSimulation.ss['node'].<property_name>['<node_name>']`

- 'demand': demand flowrate
- 'head': pressure head
- 'pressure': pressure at node
- 'elevation': height in meters with respect to DATUM
- 'type': type as defined by EPANET
- 'degree': number of neighbors
- 'processor': ID of processor in charge of computing the node
- 'leak_coefficient': emitter coefficient as defined by EPANET
- 'demand_coefficient': discharge coefficient associated with demand at the node

**Pipe properties**: can be extracted through `PTSNETSimulation.ss['pipe'].<property_name>['<pipe_name>']`
- 'start_node': name of upstream node
- 'end_node': name of downstream node
- 'length'
- 'diameter'
- 'area': cross-section area
- 'wave_speed': adjusted wave speed
- 'desired_wave_speed': wave speed set by the user (changes are reflected after initialization only)
- 'wave_speed_adjustment': relative wave speed error
- 'segments': number of segments after the water network is discretized
- 'flowrate'
- 'velocity'
- 'head_loss'
- 'direction': flow direction is 1 if water flows from upstream to downstream node, -1 otherwise
- 'ffactor': Darcy's friction factor
- 'dx': segment length
- 'type': as defined by EPANET for pipes
- 'is_inline': True if the pipe has no dead ends

**Pump properties**: can be extracted through `PTSNETSimulation.ss['pump'].<property_name>['<pump_name>']`

- 'start_node': upstream node
- 'end_node': downstream node
- 'flowrate'
- 'velocity'
- 'head_loss'
- 'direction':flow direction is 1 if water flows from upstream to downstream node, -1 otherwise
- 'initial_status': as defined in EPANET
- 'is_inline': True if the pipe has no dead ends
- 'source_head': head at upstream node
- 'a1': constant in pump equation
- 'a2': constant in pump equation
- 'Hs': constant in pump equation
- 'curve_index': index of pump curve (for internal use only)
- 'setting': percentage of nominal velocity, 1 means the pump is working a nominal capacity, 0 means the pump's velocity is zero.

**Valve properties**: can be extracted through `PTSNETSimulation.ss['valve'].<property_name>['<valve_name>']`

- 'start_node': upstream node
- 'end_node': downstream node
- 'diameter'
- 'area': cross-section area
- 'head_loss'
- 'flowrate'
- 'velocity'
- 'direction': flow direction is 1 if water flows from upstream to downstream node, -1 otherwise
- 'initial_status': as defined by EPANET
- 'type': as defined by EPANET
- 'is_inline': True if the valve has no dead ends
- 'K': discharge coefficient
- 'setting': opening percentage 1 means fully open, 0 fully closed
- 'curve_index': index of the valve curve (for internal use only)

**Open surge protection properties**: can be extracted through `PTSNETSimulation.ss['open_protection'].<property_name>['<open_protection_name>']`

- 'node': node where the protection is located
- 'area': tank cross-section area
- 'QT': tank inflow
- 'HT0': tank head at time t
- 'HT1': tank head at time t + tau

**Closed surge protection properties**: can be extracted through `PTSNETSimulation.ss['closed_protection'].<property_name>['<closed_protection_name>']`

- 'node': node where the tank is located
- 'area': tank cross-section area
- 'height'
- 'water_level': height of water level inside of the surge tank
- 'C': ideal gas constant
- 'QT0': tank inflow at time t
- 'QT1': tank inflow at time t + tau
- 'HT0': tank head at time t
- 'HT1': tank head at time t + tau
- 'HA': air head
- 'VA': air volume
