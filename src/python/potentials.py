# src/python/potentials.py

"""
@file potentials.py
@brief 定义不同类型的相互作用势能模块。
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class Potential(ABC):
    """
    @class Potential
    @brief 基类，定义相互作用势能的接口。

    属性:
        parameters (Dict[str, float]): 势能参数字典。
        cutoff (float): 截断半径。
        type (str): 势能类型。
    """

    def __init__(self, parameters: Dict[str, float], cutoff: float):
        """
        @brief 初始化一个 Potential 实例。

        @param parameters 势能参数字典。
        @param cutoff 截断半径。
        """
        self.parameters: Dict[str, float] = parameters
        self.cutoff: float = cutoff
        self.type: str = "Base"

    @abstractmethod
    def calculate_potential(self, r: float) -> float:
        """
        @brief 计算在距离 r 处的势能。

        @param r 距离。
        @return float 势能值。
        """
        pass

    @abstractmethod
    def derivative_potential(self, r: float) -> float:
        """
        @brief 计算在距离 r 处的势能导数。

        @param r 距离。
        @return float 势能导数值。
        """
        pass


class EAMPotential(Potential):
    """
    @class EAMPotential
    @brief EAM（嵌入原子模型）势能实现类。

    属性:
        parameters (Dict[str, float]): EAM势能参数。
        cutoff (float): 截断半径。
    """

    def __init__(self, parameters: Dict[str, float], cutoff: float):
        """
        @brief 初始化一个 EAMPotential 实例。

        @param parameters EAM势能参数。
        @param cutoff 截断半径。
        """
        super().__init__(parameters, cutoff)
        self.type: str = "EAM"

    def calculate_potential(self, r: float) -> float:
        """
        @brief 计算在距离 r 处的EAM势能。

        @param r 距离。
        @return float EAM势能值。
        """
        # 实现EAM势能计算逻辑
        raise NotImplementedError("EAMPotential.calculate_potential 尚未实现。")

    def derivative_potential(self, r: float) -> float:
        """
        @brief 计算在距离 r 处的EAM势能导数。

        @param r 距离。
        @return float EAM势能导数值。
        """
        # 实现EAM势能导数计算逻辑
        raise NotImplementedError("EAMPotential.derivative_potential 尚未实现。")


class LennardJonesPotential(Potential):
    """
    @class LennardJonesPotential
    @brief Lennard-Jones势能实现类。

    属性:
        parameters (Dict[str, float]): Lennard-Jones势能参数。
        cutoff (float): 截断半径。
    """

    def __init__(self, parameters: Dict[str, float], cutoff: float):
        """
        @brief 初始化一个 LennardJonesPotential 实例。

        @param parameters Lennard-Jones势能参数。
        @param cutoff 截断半径。
        """
        super().__init__(parameters, cutoff)
        self.type: str = "LennardJones"

    def calculate_potential(self, r: float) -> float:
        """
        @brief 计算在距离 r 处的Lennard-Jones势能。

        @param r 距离。
        @return float Lennard-Jones势能值。
        """
        epsilon: float = self.parameters["epsilon"]
        sigma: float = self.parameters["sigma"]
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    def derivative_potential(self, r: float) -> float:
        """
        @brief 计算在距离 r 处的Lennard-Jones势能导数。

        @param r 距离。
        @return float Lennard-Jones势能导数值。
        """
        epsilon: float = self.parameters["epsilon"]
        sigma: float = self.parameters["sigma"]
        return 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
