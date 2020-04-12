
class Workspace:
    def __init__(self, name, ic, num_nodes, num_pipes_start, num_pipes_end, time_steps):
        self.name = name
        self.results = {}
        self.num_nodes = num_nodes
        self.num_pipes_start = num_pipes_start
        self.num_pipes_end = num_pipes_end
        self.time_steps = time_steps
        self.ic = ic

    def allocate_nodes(self):
        if self.num_nodes > 0:
            self.results['node'] = Table2D(NODE_RESULTS, self.num_nodes, self.time_steps,
                labels = self.ic['node'].labels[self.where.nodes['all_to_points',]])

