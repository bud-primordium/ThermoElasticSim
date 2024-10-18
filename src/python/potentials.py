# src/python/potentials.py

import numpy as np
from python.interfaces.cpp_interface import CppInterface


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
        num_atoms = len(cell.atoms)
        positions = np.array(
            [atom.position for atom in cell.atoms], dtype=np.float64
        ).flatten()
        forces = np.zeros_like(positions)
        box_lengths = cell.get_box_lengths()
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
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[3 * i : 3 * i + 3]

    def calculate_energy(self, cell):
        num_atoms = len(cell.atoms)
        positions = np.array(
            [atom.position for atom in cell.atoms], dtype=np.float64
        ).flatten()
        box_lengths = cell.get_box_lengths()
        energy = self.cpp_interface.calculate_energy(
            num_atoms,
            positions,
            self.epsilon,
            self.sigma,
            self.cutoff,
            box_lengths,
        )
        return energy
