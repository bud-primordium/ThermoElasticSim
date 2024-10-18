# src/python/md_simulator.py


class MDSimulator:
    """
    @class MDSimulator
    @brief 分子动力学模拟器
    """

    def __init__(self, cell, potential, integrator, thermostat=None, barostat=None):
        self.cell = cell
        self.potential = potential
        self.integrator = integrator
        self.thermostat = thermostat
        self.barostat = barostat  # 为 NPT 系综预留

    def run(self, steps, dt, data_collector=None):
        # 初始化力
        self.potential.calculate_forces(self.cell)
        for step in range(steps):
            self.integrator.integrate(self.cell, self.potential, dt)
            # 应用恒温器
            if self.thermostat is not None:
                self.thermostat.apply(self.cell.atoms, dt)
            # 应用压强控制器（如果有）
            if self.barostat is not None:
                self.barostat.apply(self.cell, dt)
            if data_collector is not None:
                data_collector.collect(self.cell)
            print(f"MD Step {step} completed.")
