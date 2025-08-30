#!/usr/bin/env python3
r"""
ThermoElasticSim - Lennard-Jones 势模块

.. moduleauthor:: Gilbert Young

Lennard–Jones (12–6) 势用于近似描述惰性原子间的范德华作用：

.. math::
   V(r) = 4\,\varepsilon\Big[\Big(\frac{\sigma}{r}\Big)^{12} - \Big(\frac{\sigma}{r}\Big)^6\Big]

其中 :math:`\varepsilon` 为势阱深度（eV），:math:`\sigma` 为零势能点对应长度（Å）。

References
----------
- J. E. Jones (1924), On the Determination of Molecular Fields.
  I. From the Variation of the Viscosity of a Gas with Temperature.
  Proceedings of the Royal Society A, 106(738), 441–462. doi:10.1098/rspa.1924.0081
"""

import logging

import numpy as np

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.interfaces.cpp_interface import CppInterface
from thermoelasticsim.utils.utils import NeighborList

from .base import Potential

logger = logging.getLogger(__name__)


class LennardJonesPotential(Potential):
    r"""Lennard–Jones (12–6) 对势实现。

    Parameters
    ----------
    epsilon : float
        势阱深度 epsilon（eV）。
    sigma : float
        零势能点对应长度 sigma（Å）。
    cutoff : float
        截断距离（Å）。

    Notes
    -----
    - 势函数见模块 References 中的 Jones (1924)。
    - 单位：能量 eV，长度 Å，力 eV/Å。
    """

    def __init__(self, epsilon: float, sigma: float, cutoff: float):
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.cpp_interface = CppInterface("lennard_jones")
        logger.debug(
            f"Lennard-Jones Potential initialized with epsilon={epsilon}, sigma={sigma}, cutoff={cutoff}."
        )

    def calculate_forces(self, cell: Cell, neighbor_list: NeighborList) -> None:
        """计算作用力并写入 :code:`cell.atoms[i].force` （eV/Å）。

        Parameters
        ----------
        cell : Cell
            晶胞与原子集合。
        neighbor_list : NeighborList
            预先构建的邻居列表。
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
        """计算系统总势能（eV）。

        Parameters
        ----------
        cell : Cell
            晶胞与原子集合。
        neighbor_list : NeighborList
            预先构建的邻居列表。

        Returns
        -------
        float
            系统总势能（eV）。
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
