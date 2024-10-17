# src/python/md.py

import numpy as np
from .interfaces.cpp_interface import CppInterface


class Integrator:
    """
    @class Integrator
    @brief 积分器基类
    """

    def integrate(self, cell, potential, thermostat, dt):
        raise NotImplementedError


class VelocityVerletIntegrator(Integrator):
    """
    @class VelocityVerletIntegrator
    @brief 速度 Verlet 积分器的实现
    """

    def integrate(self, cell, potential, thermostat, dt):
        atoms = cell.atoms
        # 第一半步：更新位置
        for atom in atoms:
            atom.position += atom.velocity * dt + 0.5 * atom.force / atom.mass * dt**2
            # 应用周期性边界条件
            atom.position = cell.apply_periodic_boundary(atom.position)
        # 保存旧的力
        forces_old = [atom.force.copy() for atom in atoms]
        # 计算新的力
        potential.calculate_forces(cell)
        # 第二半步：更新速度
        for atom, force_old in zip(atoms, forces_old):
            atom.velocity += 0.5 * (atom.force + force_old) / atom.mass * dt
        # 应用恒温器
        if thermostat is not None:
            thermostat.apply(atoms, dt)


class Thermostat:
    """
    @class Thermostat
    @brief 恒温器基类
    """

    def apply(self, atoms, dt):
        raise NotImplementedError


class NoseHooverThermostat(Thermostat):
    """
    @class NoseHooverThermostat
    @brief Nose-Hoover 恒温器的实现
    """

    def __init__(self, target_temperature, time_constant):
        self.target_temperature = target_temperature
        self.Q = time_constant  # 热浴质量参数
        self.xi = 0.0  # 热浴变量初始值
        self.cpp_interface = CppInterface("nose_hoover")

    def apply(self, atoms, dt):
        num_atoms = len(atoms)
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).flatten()
        # 调用 C++ 函数
        self.xi = self.cpp_interface.nose_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            self.xi,
            self.Q,
            self.target_temperature,
        )
        # 更新原子速度
        for i, atom in enumerate(atoms):
            atom.velocity = velocities[3 * i : 3 * i + 3]


class MDSimulator:
    """
    @class MDSimulator
    @brief 分子动力学模拟器
    """

    def __init__(self, cell, potential, integrator, thermostat=None):
        self.cell = cell
        self.potential = potential
        self.integrator = integrator
        self.thermostat = thermostat

    def run(self, steps, dt, data_collector=None):
        # 初始化力
        self.potential.calculate_forces(self.cell)
        for step in range(steps):
            self.integrator.integrate(self.cell, self.potential, self.thermostat, dt)
            if data_collector is not None:
                data_collector.collect(self.cell)
            print(f"MD Step {step} completed.")
