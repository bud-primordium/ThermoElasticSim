# 文件名: thermostats.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现分子动力学模拟中的 Nose-Hoover 和 Nose-Hoover 链恒温器。

"""
恒温器模块

包含 Thermostat 基类及 Nose-Hoover 和 Nose-Hoover 链恒温器的实现
"""

import numpy as np
from .interfaces.cpp_interface import CppInterface


class Thermostat:
    """
    恒温器基类，定义恒温器的接口
    """

    def apply(self, atoms, dt):
        """应用恒温器，更新原子速度"""
        raise NotImplementedError


class NoseHooverThermostat(Thermostat):
    """
    Nose-Hoover 恒温器的实现

    Parameters
    ----------
    target_temperature : float
        目标温度
    time_constant : float
        时间常数，控制热浴耦合强度
    Q : float, optional
        热浴质量，默认为 10.0
    """

    def __init__(self, target_temperature, time_constant, Q=10.0):
        self.target_temperature = target_temperature
        self.time_constant = time_constant
        self.cpp_interface = CppInterface("nose_hoover")
        self.xi = np.array([0.0], dtype=np.float64)  # 初始热浴变量
        self.Q = Q

    def apply(self, atoms, dt):
        """
        应用 Nose-Hoover 恒温器，更新原子速度

        Parameters
        ----------
        atoms : list of Atom
            原子对象列表
        dt : float
            时间步长
        """
        num_atoms = len(atoms)
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).flatten()

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
            self.xi[0] = updated_xi
        except OSError as e:
            print(f"C++ Nose-Hoover 调用失败: {e}")
            raise

        # 更新原子速度
        velocities = velocities.reshape((num_atoms, 3))
        for i, atom in enumerate(atoms):
            atom.velocity = velocities[i]


class NoseHooverChainThermostat(Thermostat):
    """
    Nose-Hoover 链恒温器的实现

    Parameters
    ----------
    target_temperature : float
        目标温度
    time_constant : float
        时间常数
    chain_length : int, optional
        链的长度，默认为 2
    """

    def __init__(self, target_temperature, time_constant, chain_length=2):
        self.target_temperature = target_temperature
        self.time_constant = time_constant
        self.chain_length = chain_length
        self.cpp_interface = CppInterface("nose_hoover_chain")

    def apply_thermostat(self, cell, dt, xi_chain, Q):
        """
        应用 Nose-Hoover 链恒温器，更新原子速度

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        dt : float
            时间步长
        xi_chain : numpy.ndarray
            热浴变量链
        Q : numpy.ndarray
            热浴质量数组
        """
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
            self.chain_length,
            self.target_temperature,
        )

        # 更新原子速度
        velocities = velocities.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.velocity = velocities[i]
