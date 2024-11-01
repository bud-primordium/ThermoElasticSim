# 文件名: barostats.py
# 作者: Gilbert Young
# 修改日期: 2024-11-02
# 文件描述: 实现分子动力学模拟中的各种恒压器。

import numpy as np
from typing import Optional, List, Tuple
from .interfaces.cpp_interface import CppInterface
from .mechanics import StressCalculatorLJ
from .structure import Cell


class Barostat:
    """恒压器基类，定义恒压器的接口"""

    def __init__(self, target_pressure: float):
        self.target_pressure = target_pressure
        self.pressure_history = []
        self.volume_history = []
        self.stress_tensor_history = []

    def apply(self, cell, dt: float):
        """应用恒压器，更新晶胞参数"""
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
    增强版 Parrinello-Rahman-Hoover 恒压器

    Parameters
    ----------
    target_pressure : float
        目标压力 (GPa)
    time_constant : float
        控制压力调节的时间常数
    Qp : Optional[np.ndarray]
        压力热浴质量矩阵，shape=(6,)
    separate_stress_control : bool
        是否独立控制各个应力分量
    """

    def __init__(
        self,
        target_pressure: float,
        time_constant: float,
        Qp: Optional[np.ndarray] = None,
        separate_stress_control: bool = False,
    ):
        super().__init__(target_pressure)
        self.time_constant = time_constant
        self.separate_stress_control = separate_stress_control

        # 初始化压力热浴质量
        if Qp is None:
            self.Qp = np.ones(6) * (time_constant**2)
        else:
            self.Qp = Qp

        # 初始化热浴变量
        self.xi = np.zeros(6)
        self.cpp_interface = CppInterface("parrinello_rahman_hoover")

        # 用于各向异性控制
        if separate_stress_control:
            self.target_stress = np.diag([target_pressure] * 3)

    def calculate_stress_difference(self, stress_tensor: np.ndarray) -> np.ndarray:
        """计算应力差异"""
        if self.separate_stress_control:
            return stress_tensor - self.target_stress
        else:
            P_int = self.calculate_internal_pressure(stress_tensor)
            return np.full_like(stress_tensor, P_int - self.target_pressure)

    def apply(self, cell, dt: float):
        """
        应用 PRH 恒压器

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        dt : float
            时间步长
        """
        # 准备数据
        num_atoms = len(cell.atoms)
        masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
        velocities = np.array(
            [atom.velocity for atom in cell.atoms], dtype=np.float64
        ).flatten()
        forces = np.array(
            [atom.force for atom in cell.atoms], dtype=np.float64
        ).flatten()
        lattice_vectors = cell.lattice_vectors.flatten()

        try:
            # 调用 C++ PRH 函数
            self.cpp_interface.parrinello_rahman_hoover(
                dt,
                num_atoms,
                masses,
                velocities,
                forces,
                lattice_vectors,
                self.xi,
                self.Qp,
                self.target_pressure,
            )

            # 更新晶格矢量和原子速度
            cell.lattice_vectors = lattice_vectors.reshape((3, 3))
            cell.update_volume()  # 更新体积

            for i, atom in enumerate(cell.atoms):
                atom.velocity = velocities[3 * i : 3 * i + 3]

            # 记录状态
            stress_tensor = cell.calculate_stress_tensor()
            current_pressure = self.calculate_internal_pressure(stress_tensor)
            self.record_state(current_pressure, cell.Volume, stress_tensor)

        except Exception as e:
            raise RuntimeError(f"PRH barostat error: {e}")


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
