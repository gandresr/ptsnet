from phammer.simulation.funcs import run_interior_step, run_boundary_step, run_valve_step, run_pump_step

def run_step(self):
    if not self.settings.is_initialized:
        raise NotImplementedError("it is necessary to initialize the simulation before running it")
    if not self.settings.updated_settings:
        self._update_settings()

    t1 = self.t % 2; t0 = 1 - t1

    Q0 = self.mem_pool_points.flowrate[:,t0]
    H0 = self.mem_pool_points.head[:,t0]
    Q1 = self.mem_pool_points.flowrate[:,t1]
    H1 = self.mem_pool_points.head[:,t1]

    run_interior_step(
        Q0, H0, Q1, H1,
        self.point_properties.B,
        self.point_properties.R,
        self.point_properties.Cp,
        self.point_properties.Bp,
        self.point_properties.Cm,
        self.point_properties.Bm,
        self.point_properties.has_plus,
        self.point_properties.has_minus)
    run_boundary_step(
        H0, Q1, H1,
        self.node_results.leak_flow[:,self.t],
        self.node_results.demand_flow[:,self.t],
        self.point_properties.Cp,
        self.point_properties.Bp,
        self.point_properties.Cm,
        self.point_properties.Bm,
        self.ic['node'].leak_coefficient,
        self.ic['node'].demand_coefficient,
        self.ic['node'].elevation,
        self.where)
    run_valve_step(
        Q1, H1,
        self.point_properties.Cp,
        self.point_properties.Bp,
        self.point_properties.Cm,
        self.point_properties.Bm,
        self.ic['valve'].setting,
        self.ic['valve'].K,
        self.ic['valve'].area,
        self.where)
    run_pump_step(
        self.ic['pump'].source_head,
        Q1, H1,
        self.point_properties.Cp,
        self.point_properties.Bp,
        self.point_properties.Cm,
        self.point_properties.Bm,
        self.ic['pump'].a1,
        self.ic['pump'].a2,
        self.ic['pump'].Hs,
        self.ic['pump'].setting,
        self.where)
    self.pipe_results.inflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['are_dboundaries'], t1]
    self.pipe_results.outflow[:,self.t] = self.mem_pool_points.flowrate[self.where.points['are_uboundaries'], t1]
    self.node_results.head[self.where.nodes['to_points',], self.t] = self.mem_pool_points.head[self.where.nodes['to_points'], t1]
    self.t += 1
