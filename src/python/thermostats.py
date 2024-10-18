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

    def apply_thermostat(self, cell, dt, xi, Q):
        num_atoms = len(cell.atoms)
        masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in cell.atoms], dtype=np.float64
        ).flatten()
        forces = np.array(
            [atom.force for atom in cell.atoms], dtype=np.float64
        ).flatten()

        updated_xi = self.cpp_interface.nose_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            xi,
            Q,
            self.target_temperature,
        )

        # 更新原子的速度
        velocities = velocities.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.velocity = velocities[i]

        return updated_xi


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
