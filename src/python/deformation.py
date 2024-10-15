# src/python/deformation.py

"""
@file deformation.py
@brief 施加应变和变形晶胞的模块。
"""

import numpy as np
from typing import List
from structure import Cell


class Deformer:
    """
    @class Deformer
    @brief 生成和应用变形梯度矩阵的类。
    """

    def __init__(self) -> None:
        """
        @brief 初始化 Deformer 实例。
        """
        pass

    def generate_deformations(self, delta: float = 0.01) -> List[np.ndarray]:
        """
        @brief 生成六种独立的应变矩阵（3拉伸，3剪切）。

        @param delta 应变量。
        @return List[numpy.ndarray]: 变形梯度矩阵列表。
        """
        strains = [
            np.array([[1 + delta, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1 + delta, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1 + delta]]),
            np.array([[1.0, delta, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, delta], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, delta], [0.0, 0.0, 1.0]]),
        ]
        return strains

    def apply_deformation(self, cell_structure: "Cell", F: np.ndarray) -> "Cell":
        """
        @brief 应用变形梯度矩阵 F 到晶体结构。

        @param cell_structure Cell 实例。
        @param F 变形梯度矩阵，形状为 (3, 3)。
        @return Cell 变形后的晶体结构实例。
        """
        cell_structure.apply_deformation(F)
        return cell_structure
