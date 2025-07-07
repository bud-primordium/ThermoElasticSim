#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThermoElasticSim - Tersoff 势模块

.. moduleauthor:: Gilbert Young
.. created:: 2025-07-07
.. modified:: 2025-07-07
.. version:: 4.0.0
"""

import logging
from .base import Potential
from thermoelasticsim.core.structure import Cell
from thermoelasticsim.utils.utils import NeighborList

logger = logging.getLogger(__name__)

class TersoffPotential(Potential):
    """
    Tersoff 多体势的框架。

    这是一个占位符，用于未来实现 Tersoff 势，常用于模拟共价键材料如硅和碳。

    Args:
        parameters (dict): Tersoff 势的参数。
        cutoff (float): 截断距离。
    """
    def __init__(self, parameters: dict, cutoff: float):
        super().__init__(parameters, cutoff)
        logger.warning("Tersoff a势尚未完全实现，这是一个占位符。")

    def calculate_forces(self, cell: Cell, neighbor_list: NeighborList) -> None:
        """
        计算Tersoff势下的力（尚未实现）。
        """
        raise NotImplementedError("Tersoff a势的力计算尚未实现。")

    def calculate_energy(self, cell: Cell, neighbor_list: NeighborList) -> float:
        """
        计算Tersoff势下的能量（尚未实现）。
        """
        raise NotImplementedError("Tersoff势的能量计算尚未实现。")
