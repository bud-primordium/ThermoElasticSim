# src/python/structure.py

import numpy as np
from .utils import AMU_TO_EVFSA2
import logging

# 配置日志记录
logger = logging.getLogger(__name__)


class Atom:
    """
    @class Atom
    @brief 原子类，包含原子的信息和属性。
    """

    def __init__(self, id, symbol, mass_amu, position, velocity=None):
        self.id = id
        self.symbol = symbol
        self.mass_amu = mass_amu  # 保留原始质量
        self.mass = mass_amu * AMU_TO_EVFSA2  # 质量转换为 eV/fs^2
        self.position = np.array(position, dtype=np.float64)
        self.velocity = (
            np.zeros(3, dtype=np.float64)
            if velocity is None
            else np.array(velocity, dtype=np.float64)
        )
        self.force = np.zeros(3, dtype=np.float64)

    def update_position(self, delta_r):
        self.position += delta_r

    def update_velocity(self, delta_v):
        self.velocity += delta_v

    def copy(self):
        """
        @brief 创建 Atom 的深拷贝。
        @return Atom 对象的拷贝。
        """
        return Atom(
            id=self.id,
            symbol=self.symbol,
            mass_amu=self.mass_amu,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
        )


class Cell:
    """
    @class Cell
    @brief 晶胞类，包含晶格矢量和原子列表。
    """

    def __init__(self, lattice_vectors, atoms, pbc_enabled=True):
        self.lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        self.atoms = atoms  # 原子列表
        self.volume = self.calculate_volume()
        self.pbc_enabled = pbc_enabled

    def calculate_volume(self):
        return np.linalg.det(self.lattice_vectors)

    def get_box_lengths(self):
        """
        返回模拟盒子在 x、y、z 方向的长度。
        """
        box_lengths = np.linalg.norm(self.lattice_vectors, axis=1)
        return box_lengths

    def apply_deformation(self, deformation_matrix):
        """
        @brief 对晶胞和原子坐标施加变形矩阵。

        @param deformation_matrix 3x3 变形矩阵
        """
        # 更新晶格矢量
        self.lattice_vectors = np.dot(self.lattice_vectors, deformation_matrix.T)
        logger.debug(f"Updated lattice vectors:\n{self.lattice_vectors}")

        # 更新原子坐标并应用 PBC（如果启用）
        for atom in self.atoms:
            original_position = atom.position.copy()
            atom.position = np.dot(deformation_matrix, atom.position)
            if self.pbc_enabled:
                atom.position = self.apply_periodic_boundary(atom.position)
            logger.debug(
                f"Atom {atom.id} position changed from {original_position} to {atom.position}"
            )

    def apply_periodic_boundary(self, position):
        """
        @brief 应用周期性边界条件，将原子位置限制在晶胞内。

        @param position 原子的笛卡尔坐标位置
        @return 应用 PBC 后的笛卡尔坐标位置
        """
        if self.pbc_enabled:
            # 转换到分数坐标
            fractional = np.linalg.solve(self.lattice_vectors.T, position)
            # 确保在 [0, 1) 范围内
            fractional = fractional % 1.0
            # 转换回笛卡尔坐标
            new_position = np.dot(self.lattice_vectors.T, fractional)
            return new_position
        else:
            return position

    def copy(self):
        """
        @brief 创建 Cell 的深拷贝。

        @return Cell 对象的拷贝。
        """
        atoms_copy = [atom.copy() for atom in self.atoms]
        return Cell(self.lattice_vectors.copy(), atoms_copy, self.pbc_enabled)

    @property
    def num_atoms(self):
        """
        @property num_atoms
        @brief 返回原子数量。
        """
        return len(self.atoms)

    def get_positions(self):
        """
        @brief 获取所有原子的位置信息。

        @return numpy.ndarray, 形状为 (num_atoms, 3)
        """
        return np.array([atom.position for atom in self.atoms], dtype=np.float64)

    def get_velocities(self):
        """
        @brief 获取所有原子的速度信息。

        @return numpy.ndarray, 形状为 (num_atoms, 3)
        """
        return np.array([atom.velocity for atom in self.atoms], dtype=np.float64)

    def get_forces(self):
        """
        @brief 获取所有原子的力信息。

        @return numpy.ndarray, 形状为 (num_atoms, 3)
        """
        return np.array([atom.force for atom in self.atoms], dtype=np.float64)
