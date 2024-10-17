# src/python/structure.py

import numpy as np


class Atom:
    """
    @class Atom
    @brief 原子类，包含原子的信息和属性。
    """

    def __init__(self, id, symbol, mass, position, velocity=None):
        self.id = id
        self.symbol = symbol
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.zeros(3) if velocity is None else np.array(velocity)
        self.force = np.zeros(3)

    def update_position(self, delta_r):
        self.position += delta_r

    def update_velocity(self, delta_v):
        self.velocity += delta_v


class Cell:
    """
    @class Cell
    @brief 晶胞类，包含晶格矢量和原子列表。
    """

    def __init__(self, lattice_vectors, atoms, pbc_enabled=True):
        self.lattice_vectors = np.array(lattice_vectors)
        self.atoms = atoms  # 原子列表
        self.volume = self.calculate_volume()
        self.pbc_enabled = pbc_enabled

    def calculate_volume(self):
        return np.linalg.det(self.lattice_vectors)

    def apply_deformation(self, deformation_matrix):
        """
        @brief 对晶胞和原子坐标施加变形矩阵。

        @param deformation_matrix 3x3 变形矩阵
        """
        # 更新晶格矢量
        self.lattice_vectors = np.dot(self.lattice_vectors, deformation_matrix.T)
        # 更新原子坐标
        for atom in self.atoms:
            atom.position = np.dot(deformation_matrix, atom.position)

    def apply_periodic_boundary(self, position):
        if self.pbc_enabled:
            # 转换到分数坐标
            fractional = np.linalg.solve(self.lattice_vectors.T, position)
            # 确保在 [0, 1) 范围内
            fractional = fractional % 1.0
            # 转换回笛卡尔坐标
            return np.dot(self.lattice_vectors.T, fractional)
        else:
            return position

    def copy(self):
        atoms_copy = [
            Atom(
                atom.id,
                atom.symbol,
                atom.mass,
                atom.position.copy(),
                atom.velocity.copy(),
            )
            for atom in self.atoms
        ]
        return Cell(self.lattice_vectors.copy(), atoms_copy, self.pbc_enabled)
