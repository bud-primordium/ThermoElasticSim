# src/python/strain.py

"""
@file strain.py
@brief 计算应变张量的模块。
"""

import numpy as np
from typing import Tuple


class StrainCalculator:
    """
    @class StrainCalculator
    @brief 计算应变张量的类。
    """

    def __init__(self) -> None:
        """
        @brief 初始化 StrainCalculator 实例。
        """
        pass

    def calculate_strain(self, F: np.ndarray) -> np.ndarray:
        """
        @brief 根据变形梯度矩阵 F 计算 Green-Lagrange 应变张量，并转换为 Voigt 形式。

        @param F 变形梯度矩阵，形状为 (3, 3)。
        @return numpy.ndarray: 应变张量（Voigt 形式，形状为 (6,)）。
        """
        E = 0.5 * (np.dot(F.T, F) - np.identity(3))
        # 转换为 Voigt 形式（剪切分量乘以2）
        strain_voigt = np.array(
            [E[0, 0], E[1, 1], E[2, 2], 2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1]]
        )
        return strain_voigt
