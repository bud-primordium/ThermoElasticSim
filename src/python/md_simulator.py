# 文件名: md_simulator.py
# 作者: Gilbert Young
# 修改日期: 2024-10-19
# 文件描述: 实现分子动力学模拟器 MDSimulator 类，用于执行分子动力学模拟。

"""
分子动力学模拟器模块

包含 MDSimulator 类，用于执行分子动力学模拟，并支持恒温器和压强控制器
"""


class MDSimulator:
    """
    分子动力学模拟器类

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象
    potential : Potential
        势能对象，用于计算作用力
    integrator : Integrator
        积分器对象，用于时间推进模拟
    thermostat : Thermostat, optional
        恒温器对象，用于控制温度
    barostat : Barostat, optional
        压强控制器对象，用于控制压强
    """

    def __init__(self, cell, potential, integrator, thermostat=None, barostat=None):
        """初始化 MDSimulator 对象"""
        self.cell = cell
        self.potential = potential
        self.integrator = integrator
        self.thermostat = thermostat
        self.barostat = barostat  # 为 NPT 系综预留

    def run(self, steps, dt, data_collector=None):
        """
        运行分子动力学模拟

        Parameters
        ----------
        steps : int
            模拟步数
        dt : float
            时间步长
        data_collector : DataCollector, optional
            数据收集器，用于记录模拟数据
        """
        # 初始化力
        self.potential.calculate_forces(self.cell)
        for step in range(steps):
            # 积分更新位置和速度
            self.integrator.integrate(self.cell, self.potential, dt)
            # 应用恒温器（如果存在）
            if self.thermostat is not None:
                self.thermostat.apply(self.cell.atoms, dt)
            # 应用压强控制器（如果存在）
            if self.barostat is not None:
                self.barostat.apply(self.cell, dt)
            # 数据收集
            if data_collector is not None:
                data_collector.collect(self.cell)
            print(f"MD Step {step} completed.")
