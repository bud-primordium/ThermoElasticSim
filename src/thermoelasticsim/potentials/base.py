#!/usr/bin/env python3
"""
ThermoElasticSim - 势能基类模块

.. moduleauthor:: Gilbert Young
"""

from abc import ABC, abstractmethod

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.utils.utils import NeighborList


class Potential(ABC):
    """势能计算的抽象基类。

    提供原子间势能模型的统一接口，约定最小方法集以支持分子动力学 (MD)
    中的力与能量评估。

    Parameters
    ----------
    parameters : dict
        势能参数字典（按具体模型定义）。
    cutoff : float
        势能截断距离，单位 Å。

    Notes
    -----
    - 所有具体势能模型至少需实现 ``calculate_forces`` 与 ``calculate_energy``。
    - 单位约定：能量 eV，长度 Å，力 eV/Å。
    """

    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff

    @abstractmethod
    def calculate_forces(self, cell: Cell, neighbor_list: "NeighborList") -> None:
        """计算系统中所有原子的作用力。

        Parameters
        ----------
        cell : Cell
            晶胞与原子集合。
        neighbor_list : NeighborList
            邻居列表（按模型需要使用）。

        Notes
        -----
        - 需就地写入 :code:`cell.atoms[i].force` （单位 eV/Å）。
        - 抽象方法，必须由子类实现。
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_energy(self, cell: Cell, neighbor_list: "NeighborList") -> float:
        """计算系统的总势能。

        Parameters
        ----------
        cell : Cell
            晶胞与原子集合。
        neighbor_list : NeighborList
            邻居列表（按模型需要使用）。

        Returns
        -------
        float
            系统总势能，单位 eV。

        Notes
        -----
        抽象方法，必须由子类实现。
        """
        raise NotImplementedError
