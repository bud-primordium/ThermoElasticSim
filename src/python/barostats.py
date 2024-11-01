# 文件名: barostats.py
# 作者: Gilbert Young
# 修改日期: 2024-11-02
# 文件描述: 实现分子动力学模拟中的各种恒压器。

import numpy as np
from typing import Optional, List, Tuple
from .interfaces.cpp_interface import CppInterface
from .mechanics import StressCalculator


class Barostat:
    """恒压器基类，定义恒压器的接口"""

    def __init__(self, target_pressure: float):
        self.target_pressure = target_pressure
        self.pressure_history = []
        self.volume_history = []
        self.stress_tensor_history = []

    def apply(self, cell, potential, dt):
        """
        应用恒压器，更新晶胞参数

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算应力
        dt : float
            时间步长
        """
        raise NotImplementedError

    def calculate_internal_pressure(self, stress_tensor: np.ndarray) -> float:
        """计算内部压力（应力张量的迹的负三分之一）"""
        return -np.trace(stress_tensor) / 3.0

    def record_state(self, pressure: float, volume: float, stress_tensor: np.ndarray):
        """记录系统状态"""
        self.pressure_history.append(pressure)
        self.volume_history.append(volume)
        self.stress_tensor_history.append(stress_tensor)


class ParrinelloRahmanHooverBarostat(Barostat):
    """
    Parrinello-Rahman-Hoover (PRH) 恒压器的实现

    Parameters
    ----------
    target_pressure : array_like
        目标压力张量，3x3矩阵
    time_constant : float
        控制压力调节的时间常数，单位fs
    W : float, optional
        晶胞质量参数，如果为None则自动计算
    Q : array_like, optional
        压力热浴质量参数数组，长度为6，如果为None则自动计算
    """

    def __init__(self, target_pressure, time_constant, W=None, Q=None):
        self.target_pressure = np.asarray(target_pressure)
        self.time_constant = time_constant
        self.xi = np.zeros(6)  # 热浴变量数组
        self.cpp_interface = CppInterface("parrinello_rahman_hoover")
        self.stress_calculator = StressCalculator()

        # 设置默认参数
        if W is None:
            # 根据时间常数自动计算晶胞质量
            # W应该足够大以确保晶胞振荡周期大于time_constant
            self.W = (
                (time_constant * time_constant) * np.trace(self.target_pressure) / 3.0
            )
        else:
            self.W = W

        if Q is None:
            # 自动计算热浴质量参数
            self.Q = np.ones(6) * (time_constant * time_constant)
        else:
            self.Q = np.asarray(Q)

    def apply(self, cell, potential, dt):
        """
        应用PRH恒压器，更新晶胞参数和原子速度

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算应力
        dt : float
            时间步长
        """
        # 计算当前应力张量
        stress_components = self.stress_calculator.calculate_total_stress(
            cell, potential
        )
        total_stress = stress_components["total"]

        # 准备数据
        num_atoms = len(cell.atoms)
        masses = np.array([atom.mass for atom in cell.atoms])
        velocities = cell.get_velocities().flatten()
        forces = cell.get_forces().flatten()
        lattice_vectors = cell.lattice_vectors.flatten()

        # 调用C++ PRH实现
        self.cpp_interface.parrinello_rahman_hoover(
            dt,
            num_atoms,
            masses,
            velocities,
            forces,
            lattice_vectors,
            self.xi,
            self.Q,
            total_stress.flatten(),
            self.target_pressure.flatten(),
            self.W,
        )

        # 更新晶格矢量
        cell.lattice_vectors = lattice_vectors.reshape(3, 3)

        # 更新原子速度
        velocities = velocities.reshape(-1, 3)
        for i, atom in enumerate(cell.atoms):
            atom.velocity = velocities[i]

        # 更新体积
        cell.volume = cell.calculate_volume()


class BerendsenBarostat(Barostat):
    """
    Berendsen 恒压器实现

    Parameters
    ----------
    target_pressure : float
        目标压力 (GPa)
    tau_p : float
        压力耦合时间常数
    compressibility : float
        等温压缩系数
    """

    def __init__(
        self, target_pressure: float, tau_p: float, compressibility: float = 4.57e-5
    ):
        super().__init__(target_pressure)
        self.tau_p = tau_p
        self.compressibility = compressibility

    def apply(self, cell, dt: float):
        """应用 Berendsen 恒压器"""
        # 计算当前压力
        stress_tensor = cell.calculate_stress_tensor()
        current_pressure = self.calculate_internal_pressure(stress_tensor)

        # 计算缩放因子
        scaling_factor = (
            1.0
            - dt
            / self.tau_p
            * self.compressibility
            * (self.target_pressure - current_pressure)
        ) ** (1 / 3)

        # 更新晶格向量
        cell.lattice_vectors *= scaling_factor
        cell.update_volume()

        # 更新原子位置
        for atom in cell.atoms:
            atom.position *= scaling_factor

        # 记录状态
        self.record_state(current_pressure, cell.Volume, stress_tensor)


class AndersenBarostat(Barostat):
    """
    Andersen 恒压器实现

    Parameters
    ----------
    target_pressure : float
        目标压力 (GPa)
    mass : float
        活塞质量参数
    temperature : float
        系统温度 (K)
    """

    def __init__(self, target_pressure: float, mass: float, temperature: float):
        super().__init__(target_pressure)
        self.mass = mass
        self.temperature = temperature
        self.volume_velocity = 0.0
        self.kb = 8.617333262e-5  # Boltzmann 常数 (eV/K)

    def apply(self, cell, dt: float):
        """应用 Andersen 恒压器"""
        # 计算当前压力
        stress_tensor = cell.calculate_stress_tensor()
        current_pressure = self.calculate_internal_pressure(stress_tensor)

        # 计算体积加速度
        volume = cell.Volume
        force = 3 * volume * (current_pressure - self.target_pressure)
        force += 2 * self.temperature * self.kb / volume  # 热力学项
        acceleration = force / self.mass

        # 更新体积速度和体积
        self.volume_velocity += acceleration * dt
        volume_ratio = (volume + self.volume_velocity * dt) / volume
        scaling_factor = volume_ratio ** (1 / 3)

        # 更新晶格向量和原子位置
        cell.lattice_vectors *= scaling_factor
        cell.update_volume()

        for atom in cell.atoms:
            atom.position *= scaling_factor
            atom.velocity *= scaling_factor

        # 记录状态
        self.record_state(current_pressure, cell.Volume, stress_tensor)
