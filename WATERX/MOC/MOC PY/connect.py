import wntr

class Simulation:
    def __init__(self, inp_file):
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = self.sim.run_sim()
        self.network = self.wn.get_graph()

a = Simulation('nine_pipe.inp')
a.results.link['flowrate']