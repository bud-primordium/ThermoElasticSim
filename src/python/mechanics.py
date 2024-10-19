# src/python/mechanics.py

import numpy as np
from .utils import TensorConverter
from .interfaces.cpp_interface import CppInterface


class StressCalculator:
    """
    @class StressCalculator
    @brief 应力计算器基类。
    """

    def compute_stress(self, cell, potential):
        raise NotImplementedError


class StressCalculatorLJ(StressCalculator):
    """
    @class StressCalculatorLJ
    @brief 基于 Lennard-Jones 势的应力计算器。
    """

    def __init__(self):
        self.cpp_interface = CppInterface("stress_calculator")

    def compute_stress(self, cell, potential):
        # 计算并更新 cell 中的原子力
        potential.calculate_forces(cell)

        # 获取所需的物理量
        volume = cell.calculate_volume()
        atoms = cell.atoms
        num_atoms = len(atoms)
        positions = np.array(
            [atom.position for atom in atoms], dtype=np.float64
        ).flatten()
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = cell.get_forces().flatten()  # 从 cell 获取更新后的 forces
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        box_lengths = cell.get_box_lengths()

        # 初始化应力张量数组
        stress_tensor = np.zeros(9, dtype=np.float64)

        # 调用 C++ 实现的应力计算函数
        self.cpp_interface.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses,
            volume,
            box_lengths,
            stress_tensor,
        )

        # 重新整形为 3x3 矩阵
        stress_tensor = stress_tensor.reshape(3, 3)
        return stress_tensor


class StrainCalculator:
    """
    @class StrainCalculator
    @brief 应变计算器类。
    """

    def compute_strain(self, F):
        """
        @brief 计算应变张量。

        @param F 变形矩阵
        @return 应变向量，形状为 (6,)
        """
        strain_tensor = 0.5 * (F + F.T) - np.identity(3)  # 线性应变张量
        # 转换为 Voigt 表示
        strain_voigt = TensorConverter.to_voigt(strain_tensor)
        # 对剪切分量乘以 2
        strain_voigt[3:] *= 2
        return strain_voigt
