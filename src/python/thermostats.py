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

    def __init__(self, target_temperature, time_constant, Q=None):
        self.target_temperature = target_temperature
        self.Q = Q  # 热浴质量参数
        self.xi = 0.0  # 热浴变量初始值
        self.cpp_interface = CppInterface("nose_hoover")
        if self.Q is None:
            # 根据时间常数估计 Q
            kB = 8.617333262e-5  # eV/K
            self.Q = time_constant * time_constant * 3 * kB * target_temperature

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


class NoseHooverChainThermostat(Thermostat):
    """
    @class NoseHooverChainThermostat
    @brief Nose-Hoover 链恒温器的实现
    """

    def __init__(self, target_temperature, time_constant, chain_length=3, Q=None):
        self.target_temperature = target_temperature
        self.chain_length = chain_length
        self.Q = Q  # 热浴质量参数数组
        self.xi = np.zeros(chain_length)  # 热浴变量数组
        self.cpp_interface = CppInterface("nose_hoover_chain")
        if self.Q is None:
            # 根据时间常数估计 Q
            kB = 8.617333262e-5  # eV/K
            Q_base = time_constant * time_constant * 3 * kB * target_temperature
            self.Q = np.array([Q_base * (i + 1) for i in range(chain_length)])

    def apply(self, atoms, dt):
        num_atoms = len(atoms)
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).flatten()
        # 调用 C++ 函数
        self.xi = self.cpp_interface.nose_hoover_chain(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            self.xi,
            self.Q,
            self.chain_length,
            self.target_temperature,
        )
        # 更新原子速度
        for i, atom in enumerate(atoms):
            atom.velocity = velocities[3 * i : 3 * i + 3]
