# src/python/structure.py

"""
@file structure.py
@brief 管理晶体结构和原子信息的模块。
"""

import numpy as np
from typing import List, Optional


class Atom:
    """
    @class Atom
    @brief 表示晶体结构中的单个原子。

    属性:
        id (int): 原子的唯一标识符。
        symbol (str): 元素符号（例如 'Al', 'C'）。
        mass (float): 原子的质量。
        position (numpy.ndarray): 原子的位置向量。
        velocity (numpy.ndarray): 原子的速度向量。
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        mass: float,
        position: List[float],
        velocity: Optional[List[float]] = None,
    ):
        """
        @brief 初始化一个 Atom 实例。

        @param id 原子的唯一标识符。
        @param symbol 元素符号。
        @param mass 原子的质量。
        @param position 原子的位置向量。
        @param velocity 原子的速度向量（可选）。
        """
        self.id: int = id
        self.symbol: str = symbol
        self.mass: float = mass
        self.position: np.ndarray = np.array(position, dtype=float)
        self.velocity: np.ndarray = (
            np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)
        )

    def update_position(self, new_position: List[float]) -> None:
        """
        @brief 更新原子的位置。

        @param new_position 新的位置向量。
        """
        self.position = np.array(new_position, dtype=float)

    def update_velocity(self, new_velocity: List[float]) -> None:
        """
        @brief 更新原子的速度。

        @param new_velocity 新的速度向量。
        """
        self.velocity = np.array(new_velocity, dtype=float)


class Cell:
    """
    @class Cell
    @brief 管理晶体结构，包括晶胞参数和原子列表。

    属性:
        lattice_vectors (numpy.ndarray): 晶胞向量矩阵 (3x3)。
        atoms (List[Atom]): 原子列表。
        volume (float): 晶胞体积。
    """

    def __init__(self, lattice_vectors: List[List[float]], atoms: List[Atom]):
        """
        @brief 初始化一个 Cell 实例。

        @param lattice_vectors 晶胞向量矩阵 (3x3)。
        @param atoms 原子列表。
        """
        self.lattice_vectors: np.ndarray = np.array(
            lattice_vectors, dtype=float
        )  # 3x3 matrix
        self.atoms: List[Atom] = atoms
        self.volume: float = self.calculate_volume()

    def calculate_volume(self) -> float:
        """
        @brief 计算晶胞体积。

        @return float 晶胞体积。
        """
        return np.abs(np.linalg.det(self.lattice_vectors))

    def apply_deformation(self, F: np.ndarray) -> None:
        """
        @brief 应用变形梯度矩阵 F，更新晶胞参数和原子位置。

        @param F 变形梯度矩阵 (3x3)。
        """
        self.lattice_vectors = np.dot(F, self.lattice_vectors)
        for atom in self.atoms:
            atom.position = np.dot(F, atom.position)
        self.volume = self.calculate_volume()
