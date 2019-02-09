import wntr


class Conduit:
	def __init__(self, n, r, d, l):
		self.partitions = n
		self.roughness = r
		self.diameter = d
		self.length = l
		self.upstream_condition = None
		self.downstream_condition = None
		self.initial_conditions = []
		self.H = []
		self.V = []

	def next_step(self):
		for j in range(1,int(tf/dt)):
			t = j*dt
			tt.append(t)
			for i in range(n+1):
				# Pipe start
				if i == 0:
					self.H[i,j] = H0
					self.V[i,j] = self.V[i+1, j-1] + g/a*(H0 - self.H[i+1, j-1]) - f*dt*self.V[i+1, j-1]*abs(self.V[i+1, j-1])/(2*D)
				# Pipe end
				if i == n:
					self.V[i,j] = V0 * s[j]
					self.H[i,j] = self.H[i-1, j-1] - a/g*(self.V[i,j] - self.V[i-1,j-1]) - a/g*(f*dt*self.V[i-1,j-1]*abs(self.V[i-1,j-1])/(2*g))
				# Interior points
				if (i > 0) and (i < n):
					V1 = V[i-1,j-1]; H1 = H[i-1,j-1]
					V2 = V[i+1,j-1]; H2 = H[i+1,j-1]
					V[i,j] = 1/2*(V1 + V2 + g/a*(H1 - H2) - f*dt/(2*D)*(V1*abs(V1) + V2*abs(V2)))
					H[i,j] = 1/2*(a/g*(V1 - V2) + H1 + H2 - a/g*(f*dt/(2*D)*(V1*abs(V1) - V2*abs(V2))))

class Condition:
	
	def __init__(self):
		self.boundaries
		self.initial
		self.type


# ========================================================================================================================
# PIPE ID | LENGTH | MATERIAL (HW, DW, HW) | WAVESPEED | DIAMETER | Z0 | Z1 | NODE_1 | NODE_2 | MINOR LOSSES | dt
# ========================================================================================================================

pipes[min(pipes['length'])]
BFS Or DFS to traverse the graph and create the mesh
create massive data structure to solve iteratively the hammer equations

# In order to estimate friction the user has the possibility to choose 
# wn = wntr.network.WaterNetworkModel()
# wn.add_pattern('pat1', [1])
# wn.add_pattern('pat2', [1,2,3,4,5,6,7,8,9,10])
# wn.add_junction('node1', base_demand=0.01, demand_pattern='pat1',
# 	elevation=100.0, coordinates=(1,2))
# wn.add_junction('node2', base_demand=0.02, demand_pattern='pat2',
# 	elevation=50.0, coordinates=(1,3))
# wn.add_pipe('pipe1', 'node1', 'node2', length=304.8, diameter=0.3048, roughness=100,
# 	minor_loss=0.0, status='OPEN')
# wn.add_reservoir('res', base_head=125, head_pattern='pat1', coordinates=(0,2))
# wn.add_pipe('pipe2', 'node1', 'res', length=100, diameter=0.3048, roughness=100,
# 	minor_loss=0.0, status='OPEN')
# wn.options.time.duration = 24*3600
# wn.options.time.hydraulic_timestep = 15*60
# wn.options.time.pattern_timestep = 60*60

# wn.write_inpfile('filename.inp')

# epanet_sim = wntr.sim.EpanetSimulator(wn)
# epanet_sim_results = epanet_sim.run_sim()
# wntr_sim = wntr.sim.WNTRSimulator(wn)
# wntr_sim_results = wntr_sim.run_sim()
# Gn = wn.get_graph()

print(Gn.)