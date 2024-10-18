# src/python/potentials.py

import numpy as np
from .interfaces.cpp_interface import CppInterface


class Potential:
    """
    @class Potential
    @brief 势能基类
    """

    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff

    def calculate_potential(self, cell):
        raise NotImplementedError

    def calculate_forces(self, cell):
        raise NotImplementedError

    def calculate_energy(self, cell):
        raise NotImplementedError


class LennardJonesPotential(Potential):
    """
    @class LennardJonesPotential
    @brief Lennard-Jones 势的实现
    """

    def __init__(self, epsilon, sigma, cutoff):
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.epsilon = epsilon  # 单位 eV
        self.sigma = sigma  # 单位 Å
        self.cutoff = cutoff  # 单位 Å
        self.cpp_interface = CppInterface("lennard_jones")

    def calculate_forces(self, cell):
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        forces = np.zeros_like(positions, dtype=np.float64)
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        self.cpp_interface.calculate_forces(
            num_atoms,
            positions,
            forces,
            self.epsilon,
            self.sigma,
            self.cutoff,
            box_lengths,
        )
        # 更新原子力
        forces = forces.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[i]

    def calculate_energy(self, cell):
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        energy = self.cpp_interface.calculate_energy(
            num_atoms,
            positions,
            self.epsilon,
            self.sigma,
            self.cutoff,
            box_lengths,
        )
        return energy
