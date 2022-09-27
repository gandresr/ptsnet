## Publication Results

In this folder you will find four Jupyter Notebooks, which contain test cases for PTSNet's functionalities.

The file `1_simulate_scenarios.ipynb` demonstrates how to import PTSNet, define simulation settings, generate a transient model, set up a transient event, run a transient simulation, and get simulation results.

The file `2_get_results.ipynb` demonstrates how to load results saved in previous simulations (without the need to rerun the simulations).

The file `3_analytics.ipynb` contains examples of analytics functionalities to estimate simulation times, number of processors, and wave speed error for a specific simulation.

The file `4_SI_figures.ipynb` exeutes functions to compare PTSNet simulation results with respect to TSNet and Hammer v8i.

The folder *HAMMER* contains Hammer v8i files that were used to compare simulation results for valve closure, pump shut-off, and burst simulations on the TNET3 example.

The folder *SI_results* contains summarized results to compare transient simulations that were executed with Hammer v8i, TSNET, and PTSNET.