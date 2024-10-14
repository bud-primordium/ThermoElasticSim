# src/python/solver.py

"""
@file solver.py
@brief 求解弹性刚度矩阵 Cij 的模块。
"""

import numpy as np
from scipy.linalg import lstsq
from typing import List


class ElasticConstantSolver:
    """
    @class ElasticConstantSolver
    @brief 使用线性代数方法求解弹性刚度矩阵 Cij 的类。
    """

    def __init__(self) -> None:
        """
        @brief 初始化 ElasticConstantSolver 实例。
        """
        pass

    def solve(
        self, strain_data: List[np.ndarray], stress_data: List[np.ndarray]
    ) -> np.ndarray:
        """
        @brief 使用最小二乘法求解弹性刚度矩阵 Cij。

        @param strain_data List[numpy.ndarray]: 应变数据，形状为 (N, 6)。
        @param stress_data List[numpy.ndarray]: 应力数据，形状为 (N, 6)。
        @return numpy.ndarray: 弹性刚度矩阵 Cij，形状为 (6, 6)。
        """
        strain_matrix = np.array(strain_data)  # N x 6
        stress_matrix = np.array(stress_data)  # N x 6

        # 使用最小二乘法求解 C * strain.T = stress.T
        # 对每一应力分量独立求解
        C = np.zeros((6, 6))
        for i in range(6):
            C[i, :] = lstsq(strain_matrix, stress_matrix[:, i])[0]

        # 对称化弹性刚度矩阵
        C = 0.5 * (C + C.T)

        return C
