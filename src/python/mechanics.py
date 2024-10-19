# 文件名: mechanics.py
# 作者: Gilbert Young
# 修改日期: 2024年10月19日
# 文件描述: 实现应力和应变计算器，包括基于 Lennard-Jones 势的应力计算器。

"""
力学模块。

包含 StressCalculator 和 StrainCalculator 类，
用于计算应力和应变。
"""

import numpy as np
from .utils import TensorConverter
from .interfaces.cpp_interface import CppInterface


class StressCalculator:
    """
    应力计算器基类，定义应力计算方法的接口。
    """

    def compute_stress(self, cell, potential):
        """计算应力，需子类实现。"""
        raise NotImplementedError


class StressCalculatorLJ(StressCalculator):
    """
    基于 Lennard-Jones 势的应力计算器。

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象。
    potential : Potential
        Lennard-Jones 势能对象。
    """

    def __init__(self):
        self.cpp_interface = CppInterface("stress_calculator")

    def compute_stress(self, cell, potential):
        """
        计算 Lennard-Jones 势的应力张量。

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象。
        potential : Potential
            Lennard-Jones 势能对象。

        Returns
        -------
        numpy.ndarray
            3x3 应力张量矩阵。
        """
        # 计算并更新原子力
        potential.calculate_forces(cell)

        # 获取相关物理量
        volume = cell.calculate_volume()
        atoms = cell.atoms
        num_atoms = len(atoms)
        positions = np.array(
            [atom.position for atom in atoms], dtype=np.float64
        ).flatten()
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = cell.get_forces().flatten()  # 从 cell 获取更新后的力
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        box_lengths = cell.get_box_lengths()

        # 初始化应力张量数组
        stress_tensor = np.zeros(9, dtype=np.float64)

        # 调用 C++ 接口计算应力张量
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

        # 重新整形为 3x3 矩阵并返回
        stress_tensor = stress_tensor.reshape(3, 3)
        return stress_tensor


class StrainCalculator:
    """
    应变计算器类。

    Parameters
    ----------
    F : numpy.ndarray
        3x3 变形矩阵。
    """

    def compute_strain(self, F):
        """
        计算应变张量并返回 Voigt 表示法。

        Parameters
        ----------
        F : numpy.ndarray
            3x3 变形矩阵。

        Returns
        -------
        numpy.ndarray
            应变向量，形状为 (6,)。
        """
        strain_tensor = 0.5 * (F + F.T) - np.identity(3)  # 线性应变张量
        # 转换为 Voigt 表示法
        strain_voigt = TensorConverter.to_voigt(strain_tensor)
        # 对剪切分量乘以 2
        strain_voigt[3:] *= 2
        return strain_voigt
