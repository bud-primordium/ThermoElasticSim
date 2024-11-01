# 文件名: barostats.py
# 作者: Gilbert Young
# 修改日期: 2024-11-02
# 文件描述: 实现分子动力学模拟中的各种恒压器。

import numpy as np
from typing import Optional, List, Tuple
from .interfaces.cpp_interface import CppInterface
from .mechanics import StressCalculator
import logging

logger = logging.getLogger(__name__)


class Barostat:
    """恒压器基类，定义恒压器的接口"""

    def __init__(self, target_pressure: np.ndarray):
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
    Parrinello-Rahman-Hoover 恒压器实现

    Parameters
    ----------
    target_pressure : np.ndarray
        目标压力张量 (3x3)
    time_constant : float
        压力耦合时间常数
    compressibility : float, optional
        等温压缩系数, 默认值为4.57e-5
    W : float, optional
        晶胞质量参数, 默认为自动计算
    Q : np.ndarray, optional
        热浴质量参数数组 (9,), 默认为 ones(9) * (time_constant^2)
    stress_calculator : StressCalculator
        应力张量计算器实例
    """

    def __init__(
        self,
        target_pressure: np.ndarray,
        time_constant: float,
        compressibility: float = 4.57e-5,
        W: float = None,
        Q: np.ndarray = None,
        stress_calculator: StressCalculator = None,
    ):
        if target_pressure.shape != (3, 3):
            raise ValueError(
                f"Expected target_pressure shape (3, 3), got {target_pressure.shape}"
            )
        super().__init__(target_pressure.flatten())
        self.time_constant = time_constant
        self.compressibility = compressibility

        if Q is None:
            # 自动计算热浴质量参数为长度9的数组
            self.Q = np.ones(9) * (time_constant**2)
            logger.debug(f"Q initialized as ones(9) * (time_constant^2): {self.Q}")
        else:
            Q = np.asarray(Q, dtype=np.float64)
            if Q.shape != (9,):
                raise ValueError(f"Expected Q shape (9,), got {Q.shape}")
            self.Q = Q
            logger.debug(f"Q provided: {self.Q}")

        if W is None:
            # 重新定义 W 的计算公式，避免依赖于 target_pressure
            # 这里使用一个常见的公式：W = 1 / (compressibility * time_constant^2)    ????这他妈是咒语吧
            # 确保 W 不为零
            self.W = 1.0 / (self.compressibility * self.time_constant**2)
            logger.debug(f"W automatically calculated: {self.W}")
        else:
            self.W = W
            logger.debug(f"W provided: {self.W}")

        # 初始化 xi 为长度9的零数组
        self.xi = np.zeros(9, dtype=np.float64)
        logger.debug(f"xi initialized: {self.xi}")

        # C++ 接口初始化
        self.cpp_interface = CppInterface("parrinello_rahman_hoover")
        logger.debug("ParrinelloRahmanHooverBarostat initialized with C++ interface")

        # 应力计算器实例
        if stress_calculator is None:
            raise ValueError("StressCalculator instance must be provided.")
        self.stress_calculator = stress_calculator
        logger.debug(
            "ParrinelloRahmanHooverBarostat initialized with StressCalculator instance"
        )

    def apply(self, cell, potential, dt: float):
        """
        应用 Parrinello-Rahman-Hoover 恒压器，更新晶胞参数和原子速度

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算应力
        dt : float
            时间步长
        """
        try:
            logger.debug("Applying Parrinello-Rahman-Hoover Barostat.")
            # 计算当前应力张量
            stress_components = self.stress_calculator.calculate_total_stress(
                cell, potential
            )
            total_stress = stress_components["total"]

            # 准备数据
            num_atoms = len(cell.atoms)
            masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
            velocities = cell.get_velocities().flatten()
            forces = cell.get_forces().flatten()
            lattice_vectors = cell.lattice_vectors.flatten()
            target_pressure = self.target_pressure

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
                target_pressure.flatten(),
                self.W,
            )

            # 更新晶格矢量
            cell.lattice_vectors = lattice_vectors.reshape(3, 3)
            logger.debug(f"Updated lattice vectors:\n{cell.lattice_vectors}")

            # 更新原子速度
            velocities = velocities.reshape(-1, 3)
            for i, atom in enumerate(cell.atoms):
                atom.velocity = velocities[i]
                logger.debug(f"Atom {atom.id} updated velocity: {atom.velocity}")

            # 更新体积
            cell.volume = cell.calculate_volume()
            logger.debug(f"Updated volume: {cell.volume}")

            # 记录状态
            self.record_state(
                self.calculate_internal_pressure(total_stress.flatten()),
                cell.volume,
                total_stress.flatten(),
            )

        except Exception as e:
            logger.error(f"Error applying Parrinello-Rahman-Hoover Barostat: {e}")
            raise

    def calculate_internal_pressure(self, stress_tensor: np.ndarray) -> float:
        """
        根据应力张量计算内部压力

        Parameters
        ----------
        stress_tensor : np.ndarray
            应力张量 (9,)

        Returns
        -------
        float
            内部压力
        """
        if stress_tensor.shape != (9,):
            raise ValueError(
                f"Expected stress_tensor shape (9,), got {stress_tensor.shape}"
            )
        # 计算压力为应力张量的迹除以3
        pressure = np.trace(stress_tensor.reshape(3, 3)) / 3.0
        logger.debug(f"Calculated internal pressure: {pressure}")
        return pressure


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
        stress_tensor = cell.calculate_stress_tensor()  # 应返回 3x3 矩阵
        current_pressure = self.calculate_internal_pressure(stress_tensor.flatten())

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
        self.record_state(current_pressure, cell.volume, stress_tensor.flatten())


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
        current_pressure = self.calculate_internal_pressure(stress_tensor.flatten())

        # 计算体积加速度
        volume = cell.volume
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
        self.record_state(current_pressure, cell.volume, stress_tensor)
