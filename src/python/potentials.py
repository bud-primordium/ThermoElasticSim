# src/python/potentials.py

import numpy as np
from .interfaces.cpp_interface import CppInterface


class Potential:
    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff

    def calculate_potential(self, cell):
        raise NotImplementedError

    def calculate_forces(self, cell):
        raise NotImplementedError


class LennardJonesPotential(Potential):
    def __init__(self, epsilon, sigma, cutoff):
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.cpp_interface = CppInterface("lennard_jones")

    def calculate_forces(self, cell):
        num_atoms = len(cell.atoms)
        positions = np.array(
            [atom.position for atom in cell.atoms], dtype=np.float64
        ).flatten()
        forces = np.zeros_like(positions)
        lattice_vectors = cell.lattice_vectors.flatten()
        self.cpp_interface.calculate_forces(
            num_atoms,
            positions,
            forces,
            self.epsilon,
            self.sigma,
            self.cutoff,
            lattice_vectors,
        )
        # 更新原子力
        # 检查 forces 数组是否更新
        if np.allclose(forces, 0):
            print("警告：计算得到的力全为零")
        else:
            print("计算得到的力非零")
            print("计算得到的力数组：", forces)
        # 更新原子力
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[3 * i : 3 * i + 3]
