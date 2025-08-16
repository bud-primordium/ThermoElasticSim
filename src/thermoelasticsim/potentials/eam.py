#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThermoElasticSim - EAM 势模块

.. moduleauthor:: Gilbert Young
.. created:: 2024-11-01
.. modified:: 2025-07-07
.. version:: 4.0.0

该模块实现了嵌入式原子方法 (Embedded Atom Method, EAM) 势，
特别是基于 Mendelev et al. (2008) 参数化的铝 (Al) 势。

EAM势将系统的总能量表示为对势项和嵌入能的总和。每个原子的嵌入能
取决于其所处位置的局部电子密度，而局部电子密度则由周围原子贡献的电子密度叠加而成。

总能量表达式为：

.. math::
    E = \\sum_i F(\\rho_i) + \\frac{1}{2} \\sum_{i \\neq j} \\phi(r_{ij})

其中，:math:`F(\rho_i)` 是嵌入能函数，:math:`\rho_i` 是原子 :math:`i` 处的局部电子密度，
:math:`\phi(r_{ij})` 是原子 :math:`i` 和 :math:`j` 之间的对势函数。

局部电子密度 :math:`\rho_i` 由以下公式计算：

.. math::
    \\rho_i = \\sum_{j \\neq i} \\psi(r_{ij})

其中，:math:`\psi(r_{ij})` 是原子 :math:`j` 在原子 :math:`i` 处产生的电子密度贡献函数。

作用在原子 :math:`i` 上的力 :math:`\mathbf{F}_i` 由总能量对原子位置的负梯度给出：

.. math::
    \\mathbf{F}_i = -\\nabla_i E = -\\sum_{j \\neq i} \\left[ \\phi'(r_{ij}) + F'(\\rho_i) \\psi'(r_{ij}) + F'(\\rho_j) \\psi'(r_{ji}) \\right] \\frac{\\mathbf{r}_{ij}}{r_{ij}}

参考文献：
    Mendelev, M. I., Srolovitz, D. J., Ackland, G. J., & Asta, M. (2008).
    Development of new interatomic potentials for the Al-Mg system.
    Journal of Materials Research, 23(10), 2707-2721.

Classes:
    EAMAl1Potential: 铝的EAM势能实现。
"""

import numpy as np
import logging
from .base import Potential
from thermoelasticsim.core.structure import Cell
from thermoelasticsim.utils.utils import NeighborList
from thermoelasticsim.interfaces.cpp_interface import CppInterface

logger = logging.getLogger(__name__)


class EAMAl1Potential(Potential):
    """
    铝的嵌入式原子方法 (EAM) 势的实现。

    基于 Mendelev et al. (2008) 的参数化。

    Args:
        cutoff (float, optional): 截断距离，单位为 Å。默认为 6.5。
    """

    def __init__(self, cutoff: float = 6.5):
        parameters = {"cutoff": cutoff, "type": "Al1"}
        super().__init__(parameters=parameters, cutoff=cutoff)
        self.cpp_interface = CppInterface("eam_al1")
        logger.debug(f"EAM Al1 Potential initialized with cutoff={cutoff}.")

    def calculate_forces(self, cell: Cell, neighbor_list: NeighborList = None) -> None:
        """
        使用EAM势计算系统中所有原子的作用力。

        Args:
            cell (Cell): 包含原子信息的晶胞对象。
            neighbor_list (NeighborList, optional): 在此实现中未使用，但为保持接口一致性而保留。
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        lattice_vectors = np.ascontiguousarray(
            cell.lattice_vectors, dtype=np.float64
        ).flatten()

        forces = np.zeros_like(positions, dtype=np.float64)

        self.cpp_interface.calculate_eam_al1_forces(
            num_atoms, positions, lattice_vectors, forces
        )

        forces = forces.reshape((num_atoms, 3))
        for i, atom in enumerate(cell.atoms):
            atom.force = forces[i]

    def calculate_energy(self, cell: Cell, neighbor_list: NeighborList = None) -> float:
        """
        使用EAM势计算系统的总势能。

        Args:
            cell (Cell): 包含原子信息的晶胞对象。
            neighbor_list (NeighborList, optional): 在此实现中未使用，但为保持接口一致性而保留。

        Returns:
            float: 系统的总势能，单位为 eV。
        """
        num_atoms = cell.num_atoms
        positions = np.ascontiguousarray(
            cell.get_positions(), dtype=np.float64
        ).flatten()
        lattice_vectors = np.ascontiguousarray(
            cell.lattice_vectors, dtype=np.float64
        ).flatten()

        energy = self.cpp_interface.calculate_eam_al1_energy(
            num_atoms, positions, lattice_vectors
        )
        # logger.debug(f"Calculated EAM potential energy: {energy} eV.")  # 暂时关闭以减少输出

        return energy
