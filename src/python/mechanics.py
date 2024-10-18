# src/python/mechanics.py

import numpy as np
from .utils import TensorConverter
from .interfaces.cpp_interface import CppInterface


class StressCalculator:
    """
    @class StressCalculator
    @brief 应力计算器基类
    """

    def compute_stress(self, cell, potential):
        raise NotImplementedError


class StressCalculatorLJ(StressCalculator):
    """
    @class StressCalculatorLJ
    @brief 基于 Lennard-Jones 势的应力计算器
    """

    def __init__(self):
        self.cpp_interface = CppInterface("stress_calculator")

    def compute_stress(self, cell, potential):
        volume = cell.calculate_volume()
        atoms = cell.atoms
        num_atoms = len(atoms)
        positions = np.array(
            [atom.position for atom in atoms], dtype=np.float64
        ).flatten()
        velocities = np.array(
            [atom.velocity for atom in atoms], dtype=np.float64
        ).flatten()
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).flatten()
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        box_lengths = cell.get_box_lengths()
        # 调用 C++ 实现的应力计算函数
        stress_tensor = self.cpp_interface.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses,
            volume,
            box_lengths,
        )
        return stress_tensor
