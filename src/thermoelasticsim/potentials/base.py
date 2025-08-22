#!/usr/bin/env python3
"""
ThermoElasticSim - 势能基类模块

.. moduleauthor:: Gilbert Young
.. created:: 2024-10-14
.. modified:: 2025-07-07
.. version:: 4.0.0
"""

from abc import ABC, abstractmethod

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.utils.utils import NeighborList


class Potential(ABC):
    """
    势能计算的抽象基类，定义了所有势能模型必须遵循的接口。

    Args:
        parameters (dict): 势能相关的参数字典。
        cutoff (float): 势能的截断距离，单位为 Å。
    """

    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff

    @abstractmethod
    def calculate_forces(self, cell: Cell, neighbor_list: "NeighborList") -> None:
        """
        计算系统中所有原子的作用力。

        这是一个抽象方法，必须由子类实现。

        Args:
            cell (Cell): 包含原子位置和速度等信息的晶胞对象。
            neighbor_list (NeighborList): 用于加速计算的邻居列表对象。
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_energy(self, cell: Cell, neighbor_list: "NeighborList") -> float:
        """
        计算系统的总势能。

        这是一个抽象方法，必须由子类实现。

        Args:
            cell (Cell): 包含原子位置和速度等信息的晶胞对象。
            neighbor_list (NeighborList): 用于加速计算的邻居列表对象。

        Returns
        -------
            float: 系统的总势能，单位为 eV。
        """
        raise NotImplementedError
