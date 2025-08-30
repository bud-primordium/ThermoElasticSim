#!/usr/bin/env python3
r"""
ThermoElasticSim - Tersoff 势模块

.. moduleauthor:: Gilbert Young

Tersoff 多体势广泛用于共价材料（Si、C 等），能量包含键序依赖项以反映
成键方向性与配位影响。当前类为占位符，尚未提供数值实现。

References
----------
- J. Tersoff (1988), New empirical model for the structural properties of silicon.
  Phys. Rev. B 37, 6991. doi:10.1103/PhysRevB.37.6991
- J. Tersoff (1989), Modeling solid-state chemistry: Interatomic potentials for multicomponent systems.
  Phys. Rev. B 39, 5566. doi:10.1103/PhysRevB.39.5566
"""

import logging

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.utils.utils import NeighborList

from .base import Potential

logger = logging.getLogger(__name__)


class TersoffPotential(Potential):
    """Tersoff 多体势框架（占位符，未实现）。

    Parameters
    ----------
    parameters : dict
        Tersoff 势参数（依实现而定）。
    cutoff : float
        截断距离（Å）。

    Notes
    -----
    - 当前未实现；仅保留统一接口。
    - 参考文献见模块“References”。
    """

    def __init__(self, parameters: dict, cutoff: float):
        super().__init__(parameters, cutoff)
        logger.warning("Tersoff 势尚未实现（占位符）。")

    def calculate_forces(self, cell: Cell, neighbor_list: NeighborList) -> None:
        """计算力（未实现）。

        Raises
        ------
        NotImplementedError
            尚未实现。
        """
        raise NotImplementedError("Tersoff 势的力计算尚未实现。")

    def calculate_energy(self, cell: Cell, neighbor_list: NeighborList) -> float:
        """计算能量（未实现）。

        Raises
        ------
        NotImplementedError
            尚未实现。
        """
        raise NotImplementedError("Tersoff 势的能量计算尚未实现。")
