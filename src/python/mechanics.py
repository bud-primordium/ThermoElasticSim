# 文件名: mechanics.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现应力和应变计算器，包括基于 Lennard-Jones 势和EAM势的应力计算器。

import numpy as np
from .utils import TensorConverter
from typing import Dict
from .interfaces.cpp_interface import CppInterface
import logging

import logging
import matplotlib as mpl

# 设置matplotlib的日志级别为WARNING，屏蔽字体调试信息
mpl.set_loglevel("WARNING")

# 配置我们自己的日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressCalculator:
    """
    应力张量计算器
    处理三个张量贡献：
    1. 动能张量: p_i⊗p_i/m_i
    2. 维里张量: r_i⊗f_i
    3. 晶格应变张量: ∂U/∂h·h^T
    """

    def __init__(self):
        """初始化应力计算器"""
        self.cpp_interface = CppInterface("stress_calculator")
        logger.debug("Initialized StressCalculator with C++ interface")

    def calculate_stress_components(self, cell) -> Dict[str, np.ndarray]:
        """计算应力张量的各个组成部分"""
        try:
            # 设置输入数据
            num_atoms = len(cell.atoms)
            positions = np.ascontiguousarray(
                cell.get_positions(), dtype=np.float64
            ).flatten()
            velocities = np.ascontiguousarray(
                cell.get_velocities(), dtype=np.float64
            ).flatten()
            forces = np.ascontiguousarray(cell.get_forces(), dtype=np.float64).flatten()
            masses = np.ascontiguousarray(
                [atom.mass for atom in cell.atoms], dtype=np.float64
            )
            volume = cell.volume
            box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

            logger.debug(f"Number of atoms: {num_atoms}")
            logger.debug(f"Volume: {volume}")
            logger.debug(f"Box lengths: {box_lengths}")

            # 初始化一维应力张量数组
            stress_tensor = np.zeros(9, dtype=np.float64)
            logger.debug(f"Initial stress_tensor shape: {stress_tensor.shape}")

            # 调用接口
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

            logger.debug(
                f"After compute_stress call, stress_tensor shape: {stress_tensor.shape}"
            )
            logger.debug(f"Stress tensor content: {stress_tensor}")

            # 重塑为3x3矩阵
            stress_matrix = stress_tensor.reshape((3, 3))
            logger.debug(f"Reshaped stress matrix shape: {stress_matrix.shape}")
            logger.debug(f"Reshaped stress matrix content:\n{stress_matrix}")

            # 计算动能张量
            kinetic_tensor = np.zeros((3, 3))
            velocities = velocities.reshape((num_atoms, 3))
            for i in range(num_atoms):
                p = velocities[i] * masses[i]
                kinetic_tensor += np.outer(p, p) / masses[i]

            # 计算维里张量
            virial_tensor = np.zeros((3, 3))
            positions = positions.reshape((num_atoms, 3))
            forces = forces.reshape((num_atoms, 3))
            for i in range(num_atoms):
                virial_tensor += np.outer(positions[i], forces[i])

            # 归一化
            kinetic_tensor /= volume
            virial_tensor /= volume

            return {
                "kinetic": kinetic_tensor,
                "virial": virial_tensor,
                "total_basic": stress_matrix,
            }

        except Exception as e:
            logger.error(f"Error in stress calculation: {e}")
            logger.error(f"Current state of arrays:")
            logger.error(f"positions shape: {positions.shape}")
            logger.error(f"velocities shape: {velocities.shape}")
            logger.error(f"forces shape: {forces.shape}")
            logger.error(f"masses shape: {masses.shape}")
            raise

    def calculate_dUdh_tensor(self, cell, potential, dr: float = 1e-8) -> np.ndarray:
        """
        计算晶格应变应力张量

        Returns
        -------
        np.ndarray
            3x3晶格应变应力张量
        """
        lattice = cell.lattice_vectors
        dUdh = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                # 形变矩阵
                deformation = np.eye(3)
                deformation[i, j] += dr

                # 计算能量差
                cell_plus = cell.copy()
                cell_plus.apply_deformation(deformation)
                energy_plus = potential.calculate_energy(cell_plus)

                deformation[i, j] -= 2 * dr
                cell_minus = cell.copy()
                cell_minus.apply_deformation(deformation)
                energy_minus = potential.calculate_energy(cell_minus)

                dUdh[i, j] = (energy_plus - energy_minus) / (2 * dr)

        # 转换为应力张量
        stress_tensor = np.dot(dUdh, lattice.T) / cell.volume
        return stress_tensor

    def calculate_total_stress(self, cell, potential) -> Dict[str, np.ndarray]:
        """
        计算总应力张量及其组成

        Parameters
        ----------
        cell : Cell
            模拟晶胞对象
        potential : Potential
            势能对象

        Returns
        -------
        Dict[str, np.ndarray]
            包含各个3x3应力张量的字典
        """
        try:
            # 获取基础张量组成
            components = self.calculate_stress_components(cell)

            # 计算晶格应变张量
            lattice_stress = self.calculate_dUdh_tensor(cell, potential)

            # 组合总张量
            total_stress = components["total_basic"] + lattice_stress

            components["lattice"] = lattice_stress
            components["total"] = total_stress

            return components

        except Exception as e:
            logger.error(f"Error calculating stress tensors: {e}")
            raise

    def validate_tensor_symmetry(
        self, tensor: np.ndarray, tolerance: float = 1e-10
    ) -> bool:
        """验证应力张量是否对称"""
        return np.allclose(tensor, tensor.T, atol=tolerance)


class StressCalculatorLJ(StressCalculator):
    """
    基于 Lennard-Jones 势的应力计算器

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象
    potential : Potential
        Lennard-Jones 势能对象
    """

    def __init__(self):
        self.cpp_interface = CppInterface("stress_calculator")

    def compute_stress(self, cell, potential):
        """
        计算 Lennard-Jones 势的应力张量

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            Lennard-Jones 势能对象

        Returns
        -------
        numpy.ndarray
            3x3 应力张量矩阵
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
        stress_tensor = np.zeros((3, 3), dtype=np.float64)

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


class StressCalculatorEAM(StressCalculator):
    """
    基于 EAM 势的应力计算器

    计算EAM势下的应力张量，包括：
    1. 对势项的贡献
    2. 电子密度的贡献
    3. 嵌入能的贡献

    Parameters
    ----------
    None
    """

    def __init__(self):
        """初始化EAM应力计算器"""
        self.cpp_interface = CppInterface("stress_calculator")

    def compute_stress(self, cell, potential):
        """
        计算 EAM 势的应力张量

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : EAMAl1Potential
            EAM 势能对象

        Returns
        -------
        numpy.ndarray
            3x3 应力张量矩阵，单位为 eV/Å³

        Notes
        -----
        EAM势的应力张量计算包括：
        1. 对势部分的应力贡献
        2. 由电子密度梯度产生的应力贡献
        3. 嵌入能导致的应力贡献
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
        forces = cell.get_forces().flatten()  # 从cell获取更新后的力
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        box_lengths = cell.get_box_lengths()

        # 初始化应力张量数组
        stress_tensor = np.zeros((3, 3), dtype=np.float64)

        # 调用C++接口计算EAM应力张量
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

        # 重新整形为3x3矩阵并返回
        stress_tensor = stress_tensor.reshape(3, 3)
        return stress_tensor


class StrainCalculator:
    """
    应变计算器类

    Parameters
    ----------
    F : numpy.ndarray
        3x3 变形矩阵
    """

    def compute_strain(self, F):
        """
        计算应变张量并返回 Voigt 表示法

        Parameters
        ----------
        F : numpy.ndarray
            3x3 变形矩阵

        Returns
        -------
        numpy.ndarray
            应变向量，形状为 (6,)
        """
        strain_tensor = 0.5 * (F + F.T) - np.identity(3)  # 线性应变张量
        # 转换为 Voigt 表示法
        strain_voigt = TensorConverter.to_voigt(strain_tensor)
        # 对剪切分量乘以 2
        strain_voigt[3:] *= 2
        return strain_voigt
