# src/python/thermostats.py

import numpy as np
from .interfaces.cpp_interface import CppInterface


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
        self.time_constant = time_constant
        self.cpp_interface = CppInterface("nose_hoover")
        self.xi = 0.0  # 初始热浴变量
        self.Q = 10.0  # 热浴质量参数，可根据需要调整

    def apply(self, atoms, dt):
        num_atoms = len(atoms)
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).flatten()

        # 调用 C++ Nose-Hoover 函数
        try:
            updated_xi = self.cpp_interface.nose_hoover(
                dt,
                num_atoms,
                masses,
                velocities,
                forces,
                self.xi,
                self.Q,
                self.target_temperature,
            )
            self.xi = updated_xi
        except OSError as e:
            print(f"C++ Nose-Hoover 调用失败: {e}")
            raise

        # 更新原子的速度
        velocities = velocities.reshape((num_atoms, 3))
        for i, atom in enumerate(atoms):
            atom.velocity = velocities[i]


class NoseHooverChainThermostat(Thermostat):
    """
    @class NoseHooverChainThermostat
    @brief Nose-Hoover 链恒温器的实现
    """

    def __init__(self, target_temperature, time_constant, chain_length=2):
        self.target_temperature = target_temperature
        self.time_constant = time_constant
        self.chain_length = chain_length
        self.cpp_interface = CppInterface("nose_hoover_chain")

    def apply_thermostat(self, cell, dt, xi_chain, Q):
        num_atoms = len(cell.atoms)
        masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in cell.atoms], dtype=np.float64
        ).flatten()
        forces = np.array(
            [atom.force for atom in cell.atoms], dtype=np.float64
        ).flatten()

        self.cpp_interface.nose_hoover_chain(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi_chain,
            Q,
            self.target_temperature,
        )

        # 更新原子的速度
        velocities = velocities.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.velocity = velocities[i]
