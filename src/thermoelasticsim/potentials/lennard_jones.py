#!/usr/bin/env python3
"""
ThermoElasticSim - Lennard-Jones 势模块

.. moduleauthor:: Gilbert Young
.. created:: 2024-10-14
.. modified:: 2025-07-07
.. version:: 4.0.0
"""

import logging

import numpy as np

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.interfaces.cpp_interface import CppInterface
from thermoelasticsim.utils.utils import NeighborList

from .base import Potential

logger = logging.getLogger(__name__)


class LennardJonesPotential(Potential):
    """
    Lennard-Jones (LJ) 对势的实现。

    LJ势常用于模拟惰性气体原子间的相互作用。

    Args:
        epsilon (float): 势阱深度，单位为 eV。
        sigma (float): 零势能点对应的原子间距，单位为 Å。
        cutoff (float): 截断距离，单位为 Å。
    """

    def __init__(self, epsilon: float, sigma: float, cutoff: float):
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.cpp_interface = CppInterface("lennard_jones")
        logger.debug(
            f"Lennard-Jones Potential initialized with epsilon={epsilon}, sigma={sigma}, cutoff={cutoff}."
        )

    def calculate_forces(self, cell: Cell, neighbor_list: NeighborList) -> None:
        """
        使用LJ势计算系统中所有原子的作用力。

        Args:
            cell (Cell): 包含原子信息的晶胞对象。
            neighbor_list (NeighborList): 预先构建的邻居列表。
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        neighbor_pairs = [
            (i, j)
            for i in range(num_atoms)
            for j in neighbor_list.get_neighbors(i)
            if j > i
        ]

        neighbor_list_flat = [index for pair in neighbor_pairs for index in pair]
        neighbor_list_array = np.ascontiguousarray(neighbor_list_flat, dtype=np.int32)

        forces = np.zeros_like(positions, dtype=np.float64)

        self.cpp_interface.calculate_lj_forces(
            num_atoms,
            positions,
            forces,
            self.parameters["epsilon"],
            self.parameters["sigma"],
            self.cutoff,
            box_lengths,
            neighbor_list_array,
            len(neighbor_pairs),
        )

        forces = forces.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[i]

    def calculate_energy(self, cell: Cell, neighbor_list: NeighborList) -> float:
        """
        使用LJ势计算系统的总势能。

        Args:
            cell (Cell): 包含原子信息的晶胞对象。
            neighbor_list (NeighborList): 预先构建的邻居列表。

        Returns
        -------
            float: 系统的总势能，单位为 eV。
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        box_lengths = np.ascontiguousarray(cell.get_box_lengths(), dtype=np.float64)

        neighbor_pairs = [
            (i, j)
            for i in range(num_atoms)
            for j in neighbor_list.get_neighbors(i)
            if j > i
        ]

        neighbor_list_flat = [index for pair in neighbor_pairs for index in pair]
        neighbor_list_array = np.ascontiguousarray(neighbor_list_flat, dtype=np.int32)

        energy = self.cpp_interface.calculate_lj_energy(
            num_atoms,
            positions,
            self.parameters["epsilon"],
            self.parameters["sigma"],
            self.cutoff,
            box_lengths,
            neighbor_list_array,
            len(neighbor_pairs),
        )
        return energy
