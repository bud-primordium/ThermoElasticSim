# src/python/mechanics.py

import numpy as np
from .utils import TensorConverter
from .interfaces.cpp_interface import CppInterface


class StressCalculator:
    def compute_stress(self, cell, potential):
        raise NotImplementedError


class StressCalculatorLJ(StressCalculator):
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
        epsilon = potential.epsilon
        sigma = potential.sigma
        cutoff = potential.cutoff
        lattice_vectors = cell.lattice_vectors.flatten()
        # 调用 C++ 实现的应力计算函数
        stress_tensor_flat = self.cpp_interface.compute_stress(
            num_atoms,
            positions,
            velocities,
            forces,
            masses,
            volume,
            epsilon,
            sigma,
            cutoff,
            lattice_vectors,
        )
        return stress_tensor_flat


class StrainCalculator:
    def compute_strain(self, deformation_gradient):
        C = np.dot(deformation_gradient.T, deformation_gradient)
        strain_tensor = 0.5 * (C - np.identity(3))
        return strain_tensor


class ElasticConstantsSolver:
    def solve(self, strains, stresses):
        strains = np.array(strains)
        stresses = np.array(stresses)
        C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
        return C
