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
logging.basicConfig(level=logging.DEBUG)
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
        self.cpp_interface = CppInterface("stress_calculator")
        logger.debug("Initialized StressCalculator with C++ interface")

    def calculate_stress_components(self, cell) -> Dict[str, np.ndarray]:
        """
        计算动能和维里应力张量（基础应力张量）

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象

        Returns
        -------
        Dict[str, np.ndarray]
            包含基础应力张量的字典，键为 "total_basic"。
        """
        try:
            logger.debug("Starting basic stress components calculation.")

            # 设置输入数据
            num_atoms = len(cell.atoms)
            positions = cell.get_positions().flatten()
            velocities = cell.get_velocities().flatten()
            forces = cell.get_forces().flatten()
            masses = np.array([atom.mass for atom in cell.atoms], dtype=np.float64)
            volume = cell.volume
            box_lengths = cell.get_box_lengths().flatten()

            logger.debug(f"num_atoms: {num_atoms}")
            logger.debug(f"positions shape: {positions.shape}")
            logger.debug(f"velocities shape: {velocities.shape}")
            logger.debug(f"forces shape: {forces.shape}")
            logger.debug(f"masses shape: {masses.shape}")
            logger.debug(f"volume: {volume}")
            logger.debug(f"box_lengths: {box_lengths}")

            # 初始化应力张量数组
            stress_tensor = np.zeros(9, dtype=np.float64)

            # 调用C++接口计算基础应力张量(动能+维里)
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

            logger.debug(f"Computed basic stress_tensor: {stress_tensor}")

            # 重塑为3x3矩阵
            total_basic = stress_tensor.reshape((3, 3))
            logger.debug(f"total_basic matrix:\n{total_basic}")

            return {
                "total_basic": total_basic,
            }

        except Exception as e:
            logger.error(f"Error in stress calculation: {e}")
            raise

    def calculate_lattice_stress(self, cell, potential, dr=1e-8) -> np.ndarray:
        """
        计算晶格应变应力张量

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算能量
        dr : float, optional
            形变量的步长, 默认值为1e-8

        Returns
        -------
        np.ndarray
            晶格应变应力张量 (3x3)
        """
        try:
            logger.debug("Starting lattice stress calculation.")

            # 初始化能量导数矩阵
            dUdh = np.zeros((3, 3), dtype=np.float64)

            # 保存原始状态的深拷贝
            original_cell = cell.copy()
            original_lattice = original_cell.lattice_vectors.copy()
            original_positions = original_cell.get_positions().copy()

            logger.debug("Created deep copy of cell for deformation.")

            for i in range(3):
                for j in range(3):
                    logger.debug(f"Calculating derivative for component ({i}, {j}).")

                    # 正向形变矩阵
                    deformation = np.eye(3)
                    deformation[i, j] += dr

                    # 负向形变矩阵
                    deformation_negative = np.eye(3)
                    deformation_negative[i, j] -= dr

                    # 应用正向形变到原始_cell的副本
                    deformed_cell_plus = original_cell.copy()
                    deformed_cell_plus.apply_deformation(deformation)
                    energy_plus = potential.calculate_energy(deformed_cell_plus)
                    logger.debug(f"Energy after positive deformation: {energy_plus}")

                    # 应用负向形变到原始_cell的副本
                    deformed_cell_minus = original_cell.copy()
                    deformed_cell_minus.apply_deformation(deformation_negative)
                    energy_minus = potential.calculate_energy(deformed_cell_minus)
                    logger.debug(f"Energy after negative deformation: {energy_minus}")

                    # 计算能量导数
                    dUdh[i, j] = (energy_plus - energy_minus) / (2 * dr)
                    logger.debug(f"dUdh[{i}, {j}] = {dUdh[i, j]}")

            # 转换为应力张量
            lattice_stress = np.dot(dUdh, original_lattice.T) / original_cell.volume
            logger.debug(f"Lattice stress tensor:\n{lattice_stress}")

            return lattice_stress

        except Exception as e:
            logger.error(f"Error in lattice stress calculation: {e}")
            raise

    def calculate_total_stress(self, cell, potential) -> Dict[str, np.ndarray]:
        """
        计算总应力张量及其组成

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        potential : Potential
            势能对象，用于计算能量

        Returns
        -------
        Dict[str, np.ndarray]
            包含 "total_basic", "lattice", "total" 的应力张量字典
        """
        try:
            logger.debug("Starting total stress calculation.")

            # 获取基础张量组成(动能+维里)
            components = self.calculate_stress_components(cell)

            # 计算晶格应变张量
            lattice_stress = self.calculate_lattice_stress(cell, potential)

            # 组合总张量(基础 + 晶格)
            total_stress = components["total_basic"] + lattice_stress

            components["lattice"] = lattice_stress
            components["total"] = total_stress

            logger.debug(f"Total stress tensor:\n{total_stress}")

            return components

        except Exception as e:
            logger.error(f"Error calculating total stress tensors: {e}")
            raise

    def validate_tensor_symmetry(
        self, tensor: np.ndarray, tolerance: float = 1e-10
    ) -> bool:
        """
        验证应力张量是否对称

        Parameters
        ----------
        tensor : np.ndarray
            应力张量 (3x3)
        tolerance : float, optional
            对称性的容差, 默认值为1e-10

        Returns
        -------
        bool
            如果对称则为True, 否则为False
        """
        if tensor.shape != (3, 3):
            logger.error(f"Tensor shape is {tensor.shape}, expected (3, 3).")
            return False

        is_symmetric = np.allclose(tensor, tensor.T, atol=tolerance)
        if not is_symmetric:
            logger.warning("Stress tensor is not symmetric.")
        else:
            logger.debug("Stress tensor is symmetric.")

        return is_symmetric


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
