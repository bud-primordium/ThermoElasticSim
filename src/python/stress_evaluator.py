# src/python/stress_evaluator.py

"""
@file stress_evaluator.py
@brief 计算应力张量的模块。
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

# 假设 Fortran 接口已经通过 f2py 或 Cython 绑定到 Python
# from fortran_interface import calculate_stress_fortran


class StressEvaluator(ABC):
    """
    @class StressEvaluator
    @brief 抽象基类，定义应力计算的接口。
    """

    @abstractmethod
    def compute_stress(
        self, crystal_structure: "CrystalStructure", potential: "Potential"
    ) -> np.ndarray:
        """
        @brief 计算晶体结构的应力张量。

        @param crystal_structure CrystalStructure 实例。
        @param potential Potential 实例。
        @return numpy.ndarray: 应力张量（Voigt 形式，形状为 (6,)）。
        """
        pass


class EAMStressEvaluator(StressEvaluator):
    """
    @class EAMStressEvaluator
    @brief 使用EAM势能计算应力张量的实现类。
    """

    def compute_stress(
        self, crystal_structure: "CrystalStructure", potential: "Potential"
    ) -> np.ndarray:
        """
        @brief 使用EAM势能计算晶体结构的应力张量。

        @param crystal_structure CrystalStructure 实例。
        @param potential EAMPotential 实例。
        @return numpy.ndarray: 应力张量（Voigt 形式，形状为 (6,)）。
        """
        # 调用Fortran实现的EAM应力计算函数
        # positions = np.array([p.position for p in crystal_structure.particles])
        # stress_voigt = calculate_stress_fortran(positions, crystal_structure.volume, potential.parameters, potential.cutoff)
        # return stress_voigt
        raise NotImplementedError("EAMStressEvaluator.compute_stress 尚未实现。")


class LennardJonesStressEvaluator(StressEvaluator):
    """
    @class LennardJonesStressEvaluator
    @brief 使用Lennard-Jones势能计算应力张量的实现类。
    """

    def compute_stress(
        self, crystal_structure: "CrystalStructure", potential: "Potential"
    ) -> np.ndarray:
        """
        @brief 使用Lennard-Jones势能计算晶体结构的应力张量。

        @param crystal_structure CrystalStructure 实例。
        @param potential LennardJonesPotential 实例。
        @return numpy.ndarray: 应力张量（Voigt 形式，形状为 (6,)）。
        """
        # 调用Fortran实现的Lennard-Jones应力计算函数
        # positions = np.array([p.position for p in crystal_structure.particles])
        # stress_voigt = calculate_stress_fortran(positions, crystal_structure.volume, potential.parameters, potential.cutoff)
        # return stress_voigt
        raise NotImplementedError(
            "LennardJonesStressEvaluator.compute_stress 尚未实现。"
        )
